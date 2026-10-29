#pragma once
#include <map>
#include <string>
#include <vector>

namespace trt_model {
/// 将已校准异常 ONNX 转为 TensorRT 8.6 可等价解释的图：外移抗锯齿缩放，并修正 opset18 归约轴；非异常模型不变。
/// @param bytes 原始 ONNX 字节，成功时替换为兼容图；原始文件不会改写。
/// @param metadata ONNX 元数据，成功时追加外部预处理约定供引擎保存。
/// @returns 识别并处理异常模型时返回 true；不支持的图或协议抛出异常。
bool prepareAnomalyOnnx(std::vector<char>& bytes, std::map<std::string, std::string>& metadata);
}
