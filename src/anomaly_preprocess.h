#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace trt_model {
/// 将BGR8图像以训练时相同的半像素抗锯齿核缩放并转换为RGB NCHW FP32；bicubic控制双三次/双线性核。
/// @param bgr 非空CV_8UC3图像。
/// @param size 模型元数据中的训练缩放尺寸。
/// @param bicubic true使用a=-0.5双三次核，false使用双线性核。
/// @param antialias 下采样时扩宽核并归一化，保持训练时抗混叠语义。
/// @returns 含3*宽*高元素的RGB平面数据，数值以255归一化，三次核可能产生少量越界值。
std::vector<float> anomalyInput(const cv::Mat& bgr, cv::Size size, bool bicubic, bool antialias);
}
