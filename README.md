# 统一 TensorRT 8.6.1.6 模型库

当前目录为独立 C++17 工程，统一支持 YOLOv5、YOLO11、YOLO26 的目标检测和实例分割，以及 AnomalyDINO、Dinomaly、EfficientAD 异常检测。三个旧工程只用作接口参考，没有修改其源码。

## 转换和部署方式

在程序内部调用 TensorRT 自带的 `nvonnxparser` 解析 ONNX，直接构建并保存原生 `.engine`；推理使用 TensorRT Runtime 加载 engine。没有手写 YOLO 网络、WTS 转换流程、ONNX Runtime 或独立 ONNX/protobuf 库依赖，也不通过外部 `trtexec` 转换。

`initModel()` 可以传入 `.engine`，也可以传入 `.onnx`。传入 ONNX 时，软件自动在同目录生成 `.trt8616.fp32.engine`；源 ONNX 更新后会重新构建。需要 FP16、指定输出路径或部署到只读模型目录时，请先调用 `onnxToEngine(onnxPath, enginePath, fp16)`，再初始化指定 engine。自动转换要求模型目录可写；模型转换可能较慢，Qt 界面在后台执行。

输入 batch、宽高、类别数从 ONNX 输入输出和元数据读取。普通 YOLO 检测/分割可从输出通道数推断类别；YOLO26 最终输出只有类别编号，必须保留 `names` 或 `class_count` 元数据。运行必需的元数据写入引擎网络名称，类别名称字典压缩为类别数，部署 engine 时无需附带 ONNX 或额外配置文件。

支持 FP32、三通道 NCHW 图像输入。动态 batch 默认采用模型 `batch` 元数据（缺失时为 1），动态宽高采用 `imgsz` 或异常模型尺寸元数据，构建为固定优化尺寸；无法从模型确定尺寸时明确报错，需重新导出带尺寸元数据或静态 ONNX。初始化时传入旧接口的类别数和 batch，正数会校验一致性，非正数表示自动读取。一次 `runModel` 输入一张图片，固定 batch 的多余槽位填零；切片按模型 batch 组批。

## 构建

默认本机路径可通过参数或 CMake CACHE 修改：

- TensorRT：`E:/env/dll/tensorRT/TensorRT-8.6.1.6`，CMake 严格检查 8.6.1.6。
- CUDA：`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8`。
- OpenCV：`D:/opencv4.5.5/opencv/build/x64/vc15/lib`。
- Qt：`D:/Qt/5.15.2/msvc2019_64`。

```powershell
.\build.ps1
.\build.ps1 -NoGui
# 将 TensorRT/CUDA/OpenCV 原生运行库也复制到输出目录：
.\build.ps1 -DeployRuntime
# 开发机直接启动界面，并自动加入上述本机依赖目录：
.\run_demo.ps1
```

脚本使用 Visual Studio x64 工具链和 Ninja，输出 `build-release` 和 `dist`。也可用 Qt Creator 打开 `CMakeLists.txt`，选择 MSVC x64 Qt Kit；测试界面布局文件为 `gui/mainwindow.ui`，可直接在 Designer 修改。主界面没有旧预处理页，也不链接 imageProcess/caliperTool。

发布文件：六个 `RaivasDefectDet/Seg{5,11,26}.dll`、`RaivasAnomalyTRT.dll`、共享底层 `trt_backend.dll`、`defect_alg_trt.exe`、`trt_engine_tool.exe`、`dist/include` 头文件和 `dist/lib` 导入库。共享后端统一管理插件初始化和日志生命周期。

运行需要匹配的 TensorRT 8.6.1.6 `nvinfer.dll`、`nvonnxparser.dll`、`nvinfer_plugin.dll`、`nvinfer_builder_resource.dll`，CUDA 11.8 的 cudart/cublas/cublasLt 及 TensorRT 发行包所需依赖，OpenCV 4.5.5、MSVC x64 Runtime；界面还需 Qt 平台插件，构建脚本通过 `windeployqt` 部署。TensorRT 10.8 生成的旧引擎需用 8.6 重新转换，不能混用同名运行库。

## YOLO 调用

六组公开头：`yolov5_det.h`、`yolov5_seg.h`、`yolo11_det.h`、`yolo11_seg.h`、`yolo26_det.h`、`yolo26_seg.h`。命名空间、类名及 C 导出沿用 `RaivasDefectDet11` / `RaivasDet11_*` 等形式，原有初始化、参数设置、检测、逐项取结果、切片和释放方法保留。YOLO5 旧 `runCutModel` 与统一 `runModelCutImage` 等价。

```cpp
#include "yolo11_seg.h"
RaivasDefectSeg11::RaivasDefectSeg11 model;
if (model.onnxToEngine("model.onnx", "model.engine", false) != 0) {
    // 通过 model.getLastError() 获取具体原因。
    return;
}
if (model.initModel("model.engine") != 0) return;
model.setPara(640, 128, 0.5);
int count = 0;
if (model.runModel(image, count) != 0) return;
for (int i = 0; i < count; ++i) {
    int classId; float score, x, y, w, h; cv::Mat mask;
    model.getDefectDetResult(i, &classId, &score, &x, &y, &w, &h, mask);
}
```

图片可为 8 位灰度、BGR、BGRA 或非连续 ROI；YOLO 使用 RGB/255、114 letterbox。结果为原图 x/y/width/height，分割 mask 为原图大小 CV_8UC1 独立副本（0/255）。检测阈值为置信度；分割保留原语义：参数阈值控制掩码，目标置信度固定 0.5。YOLO5 使用 objectness×类别分数；普通 YOLO 执行类别内 NMS（IoU 0.45），YOLO26 end2end 使用最终输出，不重复 NMS。切片结果映射回整图，重叠区域可能有重复目标。

YOLO 返回 0 成功，-1 文件/TRT/CUDA 错误，-2 参数或任务不匹配，-3 未初始化。失败清空当次结果，错误通过类 `getLastError()` 或 C 导出 `GetLastError()` 读取。同实例方法串行，不同对象使用独立上下文和缓冲区。

旧 WTS 和 YOLO11 InitArch 入口保留迁移提示并返回 -2，不再构建旧架构。旧底层网络搭建/反序列化辅助函数未公开；公开类已采用 PImpl，集成新 DLL 时调用程序必须用新头文件重新编译。原 cv::Mat 参数及 C++ 字符串对象需要匹配 OpenCV、编译器和 `/MD` 运行库，不能直接作为纯 C/C# 数据结构使用。

## 异常检测

`anomaly_trt.h` 提供 `AnomalyDINOTRT`、`DinomalyTRT`、`EfficientADTRT`，共用 `AnomalyTRT` 接口；C 导出为独立句柄 `RaivasAnomalyTRT_*`。

```cpp
#include "anomaly_trt.h"
EfficientADTRT model;
if (model.onnxToEngine("efficientad.onnx", "efficientad.engine", false) != 0) return;
if (model.initModel("efficientad.engine") != 0) return;
model.setPara(-1, 0.75); // 图片阈值用模型校准值，像素阈值设为 0.75。
int count = 0;
AnomalyResult result;
if (model.runModel(image, count) == 0) model.getResult(result);
```

接收之前异常检测工程导出的 `anomalib.contract_version=1` 已校准 ONNX，保留图内 Normalize、网络和校准逻辑。TensorRT 8.6 对 opset18 的抗锯齿 Resize 和归约轴支持存在差异：转换时通过内置 C++ 代码在内存中外移首个抗锯齿 Resize、等价转换常量归约轴，再交给 TensorRT 解析；原始 ONNX 文件不改写。这一步没有 Python/ONNX 库依赖。

兼容处理还会将带 start/end 的 Shape 转成显式 Shape+Slice，并在完整检查消费者关系后把“全量排序再取前 K”的图等价替换为 TopK(K)，绕开 TensorRT 8.6 的 TopK 上限。处理针对这三个已有导出模型的算子语义，不包含手写模型架构或 CPU 网络回退。

外移的缩放由 C++ 实现与原部署 Resize 对应的半像素线性/三次抗锯齿核；不使用 YOLO letterbox。尺寸、颜色、模型类型、归一化阈值从模型元数据获得。模型缺少协议或包含不支持的预处理时明确报错。分数、异常图、mask、热力图和叠加图均可读取，图像还原到原图尺寸。

`setPara(imageThreshold, pixelThreshold)` 接收 [0,1]，-1 使用模型校准值。异常分数是校准分数，不能视为缺陷概率；不逐图做最大最小拉伸。mask 在网络输出分辨率阈值化后用最近邻恢复，热力图只显示 mask 内区域。异常模型精度应先使用 FP32 对照验证后，再评估 FP16 对阈值附近结果的影响。

### Dinomaly 的 FP16 限制（2026-10-29）

已实测普通 FP16 使本机 Dinomaly 四张参考图片的图片分数全部变成 0。其线性注意力分母达到 `1,141,135.625`，未归一化矩阵乘积也有大量元素超过 FP16 最大值 `65504`。只把注意力、归一化和校准恢复为 FP32 后，临界缺陷仍从 FP32 的 `0.501101` 变为 `0.49785`，跨过 `0.5` 判定阈值，不能保证检出。

因此当前版本对 Dinomaly 固定使用 FP32，界面的半精度选项禁用并明确显示“精度保护”；DLL/转换工具即使收到 `fp16=true` 也会记录警告并构建 FP32。实际精度写入引擎元数据 `trt_model.effective_precision` / `trt_model.precision_policy`，可用 `trt_engine_tool inspect` 查询。其他模型的精度选择保持不变。旧 FP16 引擎不会自动修复，须重新转换；使用 Qt Creator 旧构建目录时应重新编译并重启程序。

开发回归脚本为 `tests/verify_dinomaly_precision.py`，对照普通 FP16、FP32 和保护后的引擎，不修改图片/像素阈值。该保护以可靠性为目标，不提供 Dinomaly 的 FP16 提速保证。

三个异常模型均支持 `setCutPara(cutSize, overlappingSize)` + `runCutModel(image, len)`，`runModelCutImage` 为同义接口。也可用三参 `setPara(cutSize, overlappingSize, pixelThreshold)` 同时配置切片和像素阈值。切片覆盖全图并复用当前模型，结果恢复为整图：重叠异常图逐像素取最大值、mask 做 OR、整图 score 取切片最大值，按图片阈值判定整图类别；成功仍返回 `len=1` 并通过 `getResult` 获取合并结果。C 句柄接口对应 `RaivasAnomalyTRT_setCutPara` / `RaivasAnomalyTRT_DetectCut`。

## 测试界面

启动 `dist/bin/defect_alg_trt.exe`，左侧选择九种模型任务之一，选择 ONNX 和 engine 路径，点击转换、初始化，打开图片或目录后检测。只读区显示自动识别的尺寸、batch 和类别。支持切片、连续检测、原图/叠加图/热力图/掩码切换、滚轮缩放、平移、保存当前图像及释放当前模型。模型操作在后台执行，操作期间拒绝关闭窗口以保护资源生命周期。

```powershell
# 任务编号 0..5 为 YOLO5/11/26 检测、分割，6..8 为三个异常模型。
.\dist\bin\defect_alg_trt.exe --smoke-test 2 validation/yolo11n.engine path/to/image.jpg validation/gui-yolo11
```

## 开发验证

`trt_engine_tool build model.onnx model.engine [fp16]` 使用与 DLL 相同的进程内转换实现；`inspect model.engine` 查看输入输出和元数据。此工具只便于调试，现场软件可直接调用 DLL，不需要单独启动转换工具。

`tests/make_yolo_synthetic.py` 生成六个小型测试图，`yolo_synthetic_tests validation/synthetic` 覆盖 batch=2、矩形输入、NMS、objectness、end2end、掩码、切片、错误恢复和资源释放。`tests/verify_yolo_reference.py` 对真实模型的 FP32 引擎与 ONNX 输出做数值对照；`anomaly_validation` 对异常模型的参考分数、异常图和判定做核验。Python、onnx、onnxruntime 仅用于开发机导出和数值验证，不参与 DLL 或界面运行。

`tools/export_yolo_onnx.py` 使用本机官方源码和已有 PT 导出标准静态 ONNX，不下载权重或安装依赖。YOLO26 导出会根据 YAML 恢复旧权重缺失的 end2end 属性，避免融合时删掉 one2one 分支。实际验证结果另见 `validation` 报告。

### 分段性能测试

启用测试和界面构建后，`build-release/yolo_performance.exe` 可测量纯 GPU enqueue、包含传输的执行封装、公开检测接口、界面结果图生成和单独原图复制。GPU 使用 CUDA 事件，其余使用同步墙钟；排除模型初始化、读图和 Qt 实际上屏，每段独立预热并统计中位数、平均值和 P95，不能把各段中位数直接相加。

```powershell
# 依赖目录加入 PATH 后执行，最后的120为测量次数，另有独立预热。
.\build-release\yolo_performance.exe 5 data/det/yolo5.engine data/det/2026_07_08_14_49_52_837_4.jpg 120
.\build-release\yolo_performance.exe 5 data/det/yolo516.engine data/det/2026_07_08_14_49_52_837_4.jpg 120
```

指定大图的对照结果见 `validation/yolov5-performance-report.json`：FP16 纯 GPU 推理由约 9.01 ms 降至 2.79 ms，但含原图复制/绘制的流程仅从约 33.15 ms 降至 27.79 ms。测试界面的“总耗时”包含结果图生成；约 92 MB 原图的一次复制本身约需 18～20 ms，不能用此总耗时直接判断 GPU 是否启用了半精度。
