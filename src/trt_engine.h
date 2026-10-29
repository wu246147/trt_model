#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace trt_model {
/// 内部张量描述；shape 为实际维度，values 为统一转成 FP32 的推理输出。
struct Tensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> values;
};

/// 从引擎和 ONNX 元数据恢复的模型信息，不依赖手填网络结构参数。
struct ModelInfo {
    int batch = 0, channels = 0, height = 0, width = 0;
    int classes = 0;
    bool endToEnd = false;
    std::string inputName;
    std::vector<Tensor> outputs;
    std::map<std::string, std::string> metadata;
};

/// 单实例拥有独立的执行上下文、CUDA 流和复用缓冲区；外层模型负责串行调用。
class Engine {
public:
    /// 创建空执行器，尚不申请 GPU 资源。
    Engine();
    /// 释放上下文、缓冲区及 CUDA 流，不重置 CUDA 设备。
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    /// 加载 TensorRT 8.6 引擎；传入 ONNX 时自动构建并复用同目录 FP32 缓存，失败抛出异常并清空旧模型。
    void load(const std::string& path);
    /// 返回已加载模型的信息；未初始化时抛出异常。
    const ModelInfo& info() const;
    /// 输入为与模型一致的 NCHW FP32 数据，返回所有输出；执行失败时抛出异常。
    std::vector<Tensor> infer(const std::vector<float>& input);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// 直接解析 ONNX 构建原生引擎并保存运行元数据；Dinomaly 即使请求 FP16 也固定 FP32，失败不覆盖已有引擎。
void buildEngine(const std::string& onnx, const std::string& engine, bool fp16 = true);
}
