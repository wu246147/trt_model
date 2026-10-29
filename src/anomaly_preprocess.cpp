#include "anomaly_preprocess.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trt_model {
namespace {
/// 单个输出坐标的有限支撑核，权重已按边界截断后重新归一化。
struct Kernel { int begin = 0; std::vector<float> weights; };

/// 计算Pillow/PyTorch抗锯齿缩放所用的线性或三次卷积核值。
float kernelValue(float distance, bool bicubic) {
    distance = std::abs(distance);
    if (!bicubic) return std::max(0.0f, 1.0f - distance);
    // 抗锯齿三次核采用a=-0.5，与原ONNX部署Resize的cubic_coeff_a一致。
    if (distance < 1.0f) return ((1.5f * distance - 2.5f) * distance) * distance + 1.0f;
    if (distance < 2.0f) return ((-0.5f * distance + 2.5f) * distance - 4.0f) * distance + 2.0f;
    return 0.0f;
}

/// 为一条轴计算半像素坐标和边界权重；antialias在缩小时扩大核支撑。
std::vector<Kernel> makeKernels(int source, int destination, bool bicubic, bool antialias) {
    std::vector<Kernel> kernels(static_cast<size_t>(destination));
    const float scale = static_cast<float>(source) / destination;
    const float filterScale = antialias ? std::max(scale, 1.0f) : 1.0f;
    const float support = (bicubic ? 2.0f : 1.0f) * filterScale;
    for (int x = 0; x < destination; ++x) {
        const float center = (x + 0.5f) * scale;
        auto& kernel = kernels[x];
        kernel.begin = std::max(0, static_cast<int>(center - support + 0.5f));
        const int end = std::min(source, static_cast<int>(center + support + 0.5f));
        kernel.weights.reserve(static_cast<size_t>(std::max(0, end - kernel.begin)));
        float sum = 0;
        for (int i = kernel.begin; i < end; ++i) { const float weight = kernelValue((i - center + 0.5f) / filterScale, bicubic); kernel.weights.push_back(weight); sum += weight; }
        if (sum == 0 || kernel.weights.empty()) throw std::runtime_error("异常模型缩放核为空");
        for (auto& weight : kernel.weights) weight /= sum;
    }
    return kernels;
}
}

/// 先横向再纵向进行可分离抗锯齿插值，避免OpenCV INTER_CUBIC与训练核系数不同造成异常图漂移。
std::vector<float> anomalyInput(const cv::Mat& bgr, cv::Size size, bool bicubic, bool antialias) {
    if (bgr.empty() || bgr.type() != CV_8UC3 || size.width <= 0 || size.height <= 0) throw std::runtime_error("异常模型预处理尺寸或图像格式无效");
    const auto horizontal = makeKernels(bgr.cols, size.width, bicubic, antialias);
    const auto vertical = makeKernels(bgr.rows, size.height, bicubic, antialias);
    const size_t plane = static_cast<size_t>(size.width) * size.height;
    if (plane > std::vector<float>().max_size() / 3) throw std::runtime_error("异常模型输入尺寸过大");
    std::vector<float> intermediate(static_cast<size_t>(bgr.rows) * size.width * 3), output(plane * 3);
    for (int y = 0; y < bgr.rows; ++y) {
        const auto* row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < size.width; ++x) {
            const auto& kernel = horizontal[x];
            for (int channel = 0; channel < 3; ++channel) {
                float value = 0;
                for (size_t j = 0; j < kernel.weights.size(); ++j) value += (row[kernel.begin + static_cast<int>(j)][2 - channel] / 255.0f) * kernel.weights[j];
                intermediate[(static_cast<size_t>(y) * size.width + x) * 3 + channel] = value;
            }
        }
    }
    for (int y = 0; y < size.height; ++y) {
        const auto& kernel = vertical[y];
        for (int x = 0; x < size.width; ++x) for (int channel = 0; channel < 3; ++channel) {
            float value = 0;
            for (size_t j = 0; j < kernel.weights.size(); ++j) value += intermediate[(static_cast<size_t>(kernel.begin + static_cast<int>(j)) * size.width + x) * 3 + channel] * kernel.weights[j];
            output[static_cast<size_t>(channel) * plane + static_cast<size_t>(y) * size.width + x] = value;
        }
    }
    return output;
}
}
