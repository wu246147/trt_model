# 验证结果（2026-10-28）

环境：Windows x64、MSVC 19.44、TensorRT 8.6.1.6、CUDA 11.8、OpenCV 4.5.5、Qt 5.15.2、RTX 3060。最终 CMake Release 编译、安装和 Qt 部署成功，产物位于 `dist`。验证使用 FP32 并关闭 TF32；没有对 FP16 精度作同等覆盖验证。

| 范围 | 结果 | 证据 |
| --- | --- | --- |
| YOLO5/11/26 检测及分割，六组合成 ONNX | 全部通过 | `validation/final-yolo-synthetic.log` |
| 真实 YOLO5 检测、YOLO11 检测/分割、YOLO26 检测/分割 | 五个模型与 ONNX 参考输出对照全部通过 | `validation/yolo_reference_report.json` |
| 奇数长宽图片的实例掩码映射 | 五种尺寸与官方掩码处理逐像素一致 | `validation/yolo_mask_scaling_report.json` |
| 三个异常模型，12 个正常/异常及不同尺寸样本 | 全部通过，分类全部一致 | `validation/anomaly-reference-report.json`、最终发布 DLL 复核 `validation/final-anomaly-reference.json` |
| 三个异常模型切片 | 全部通过，手动独立逐片融合误差为零，边界覆盖 100% | `validation/anomaly-cut-report.json` |
| 九个界面入口及 EfficientAD 界面切片 | 按钮集成测试全部通过，窗口/结果截图已检查 | `validation/gui-*` 下 `ui.log`、`window.png`、`result.png` |
| ONNX 自动转换、中文路径、大写扩展名、缓存复用、转换失败不覆盖旧引擎 | 全部通过 | `validation/auto-convert-*.log`、`validation/expected-build-error.log` |

异常模型图片分数和异常图保持原校准语义，异常图最大绝对误差 `5.30928e-5`，掩码像素一致率最低 `99.9993%`。切片结果与逐片单独推理后按 max/OR 合并的结果一致。

本机没有找到 YOLOv5 实例分割的真实 PT/ONNX 权重，该项已通过真实 TensorRT 合成引擎及界面链路验证，尚未完成真实训练权重的精度对照。其他八个界面入口均使用真实模型；GUI 异常检测使用 bus 图片检查操作链路，模型精度依据上述训练工程的正常/异常参考样本，而非这张界面示例图。

验证脚本中的 Python、ONNX、ONNX Runtime 仅为开发机数值对照依赖，发布的 C++ DLL、转换工具及界面均不依赖这些库。现场需部署匹配的 TensorRT/CUDA/OpenCV/MSVC 运行库；可使用 `build.ps1 -DeployRuntime` 收集原生依赖。

## Dinomaly FP16 问题复查（2026-10-29）

同一模型直接开启 FP16 后，四张参考图的图片得分均变为 0。原始 FP32 计算的线性注意力分母实测最高 `1,141,135.625`，大量元素超过 FP16 上限 `65504`。普通 FP16 引擎未通过精度回归。

两种局部混合精度方案消除了全零输出，但临界异常样本仍从 `0.501101` 降为 `0.49785`，分类由异常翻转为正常，因此未将这些实验方案用于发布版本。

发布版本对 Dinomaly 明确固定 FP32，不更改原阈值；界面禁用该任务的 FP16 选项，C++ 转换入口对半精度请求记录警告并执行 FP32 保护，其他模型精度选择不变。重新生成的引擎元数据确认 `requested_precision=fp16`、`effective_precision=fp32`、`precision_policy=dinomaly_fp32_required_v1`。

四张图回归全部通过，分数恢复为 `0 / 0.501101 / 0.598456 / 0.483994`，分类与原 FP32 基准一致。与训练参考相比异常图最大绝对误差 `1.12653e-5`，掩码一致率最低 `99.9996%`。真实按钮界面测试通过，截图确认“Dinomaly 固定 FP32（精度保护）”及缺陷检出结果。

报告：`validation/dinomaly-precision-report.json`；实验局部保护报告：`validation/dinomaly-fp16-regression.json`；界面产物：`validation/gui-dinomaly-fp32-guard/`。旧 FP16 engine 必须重新转换；当前保护不承诺 Dinomaly 的 FP16 加速。

## 用户提供 YOLOv5 大图的性能对照（2026-10-29）

使用 `data/det/yolo5.engine` 与 `data/det/yolo516.engine`，图片 `2026_07_08_14_49_52_837_4.jpg`，整图检测，阈值 0.5，无切片。图片 4096×7500 BGR，约 92 MB；两份引擎均为输入 1×3×640×640、输出 1×25200×7，两者检出数量均为 1。本次只对性能及数量进行核对，不代表完整的 FP16 精度验收。

各阶段独立预热后测量 120 次；GPU 使用 CUDA 事件，CPU/封装使用同步墙钟；不包含文件读取和 Qt 实际上屏，界面适配器阶段包含结果图释放。各阶段是独立测量，不应简单相加。

| 阶段，中位数 | FP32 | FP16 |
| --- | ---: | ---: |
| CPU 预处理 | 3.69 ms | 4.16 ms |
| GPU enqueue，不含传输 | 9.01 ms | 2.79 ms |
| 引擎执行，含传输/输出转换 | 9.88 ms | 3.59 ms |
| 公开检测接口，含预处理/解码/NMS | 13.21 ms | 7.20 ms |
| 界面适配器，含原图复制/结果绘制 | 33.15 ms | 27.79 ms |
| 单独原图复制 | 18.30 ms | 19.55 ms |

结论：该模型 GPU 段确有约 3.23 倍加速，公开检测接口约 1.83 倍；大图复制与绘制使结果图生成的总耗时仅下降约 16%。原图复制是 CPU/内存操作，与模型 FP16 无关。

报告包含图片/引擎 SHA256、P95、迭代次数和测试口径：`validation/yolov5-performance-report.json`。测试工具源码为 `tests/yolo_performance.cpp`，本次未修改生产检测算法或模型引擎。
