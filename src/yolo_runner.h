#pragma once
#include "trt_engine.h"
#include <opencv2/core.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trt_model {
/// 单个目标的原图坐标结果；内部仅缓存框内掩码，读取时再还原完整画布以降低大图常驻内存。
struct YoloResult { int classId; float score; cv::Rect box; cv::Mat mask; cv::Size canvas; };
/// 内部统一执行器，按版本解释 ONNX 输出；不同对象各自持有模型、结果和互斥锁。
class YoloRunner {
public:
    /// 创建指定 YOLO 版本及任务的执行器；构造时不分配 GPU 资源。
    YoloRunner(int version, bool segmentation);
    /// 释放本对象的模型、流和结果；不重置 CUDA 设备。
    ~YoloRunner();
    /// 使用 ONNX 解析器生成原生 engine；fp16 控制内部半精度，失败保留错误详情。
    int convert(const std::string& source, const std::string& destination, bool fp16);
    /// 加载并校验版本对应输出；classCount/batchSize 非正数时自动读取，否则必须匹配。
    int init(const std::string& path, int classCount = 0, int batchSize = 0);
    /// 验证并更新切片尺寸、重叠像素及检测置信度或分割掩码阈值。
    int setPara(int size, int overlap, double threshold);
    /// 释放本对象资源与结果；可安全重复调用。
    int dispose();
    /// 推理整图或切片，len 返回实例数；失败清空结果及 len。
    int run(cv::Mat& image, int& len, bool tiles = false);
    /// 读取目标坐标及独立掩码；下标越界或空输出指针返回 -2。
    int result(int index, int* classId, float* score, float* x, float* y, float* w, float* h, cv::Mat* mask = nullptr);
    /// 返回最近错误的独立字符串副本；成功调用会清除旧错误。
    std::string error() const;
    /// 读取已加载模型的自动参数；未初始化返回 -3。
    int modelInfo(int& batch, int& width, int& height, int& classes) const;
    /// 记录已废弃入口的说明并返回 -2，禁止将 WTS 路径当作 ONNX 静默处理。
    int retired(const std::string& message);
private:
    /// 校验 engine 输出布局和类别数，异常由 init 转换成错误码。
    void inspect();
    /// 推理一组有效切片；不足 engine batch 的输入槽位填零，结果映射到整图。
    void infer(const std::vector<cv::Mat>& images, const std::vector<cv::Point>& offsets, cv::Size canvas);
    int version_;
    bool segmentation_;
    mutable std::mutex mutex_;
    std::unique_ptr<Engine> engine_;
    std::string error_;
    std::vector<YoloResult> results_;
    std::vector<float> input_;
    int prediction_ = -1, prototype_ = -1, classes_ = 0, anchors_ = 0, channels_ = 0;
    int masks_ = 0, protoHeight_ = 0, protoWidth_ = 0;
    bool transposed_ = false, endToEnd_ = false;
    int cutSize_ = 640, overlap_ = 128;
    double threshold_ = 0.5;
};
}
