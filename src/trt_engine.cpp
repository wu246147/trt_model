#include "trt_engine.h"
#include "anomaly_onnx_prepare.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace trt_model {
namespace {
/// 保存本次 TensorRT 操作的错误详情；日志回调遵守 noexcept 约定。
class Logger final : public nvinfer1::ILogger {
public:
    /// 记录错误和警告；不将日志异常传播至 TensorRT 内部线程。
    void log(Severity severity, const char* message) noexcept override {
        try {
            if (severity > Severity::kWARNING) return;
            std::lock_guard<std::mutex> lock(mutex_);
            if (severity <= Severity::kERROR) errors_ += std::string(message) + "\n";
            std::cerr << "[TensorRT] " << message << '\n';
        } catch (...) {
            // 内存紧张时使用不分配 C++ 字符串的回退日志，同时遵守 noexcept 约定。
            std::fputs("[TensorRT] 日志记录失败\n", stderr);
        }
    }
    /// 返回已记录的错误，用于补全调用方可查询的失败原因。
    std::string errors() { std::lock_guard<std::mutex> lock(mutex_); return errors_; }
private:
    std::mutex mutex_;
    std::string errors_;
};

// 后端 DLL 在所有模型 DLL 之前装载、之后卸载，插件日志对象因此覆盖全部模型生命周期。
Logger pluginLogger;
std::once_flag pluginInitialization;

/// 通过共享后端在进程内只注册一次插件，避免插件保存指向栈上日志对象的悬空指针。
void initializePlugins() {
    std::call_once(pluginInitialization, [] {
        if (!initLibNvInferPlugins(&pluginLogger, "")) throw std::runtime_error("初始化 TensorRT 插件失败: " + pluginLogger.errors());
    });
}

/// 检查 CUDA 返回码，失败时携带操作名称抛出异常。
void checkCuda(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

/// 在持有模型的 GPU 上执行操作，退出作用域后恢复调用线程原来的设备。
class DeviceGuard {
public:
    /// target 为模型创建时的设备编号；切换失败时抛出异常。
    explicit DeviceGuard(int target) {
        checkCuda(cudaGetDevice(&previous_), "读取 CUDA 设备失败");
        if (previous_ != target) checkCuda(cudaSetDevice(target), "切换 CUDA 设备失败");
    }
    /// 恢复设备；析构不抛异常，设备错误由 CUDA 后续操作报告。
    ~DeviceGuard() { if (previous_ >= 0) cudaSetDevice(previous_); }
private:
    int previous_ = -1;
};

/// 使用 UTF-8 路径读取完整二进制文件，拒绝空文件和不完整读取。
std::vector<char> readBinary(const std::string& path) {
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("无法打开模型文件: " + path);
    auto length = file.tellg();
    if (length <= 0 || static_cast<unsigned long long>(length) > std::numeric_limits<size_t>::max())
        throw std::runtime_error("模型文件为空或过大: " + path);
    std::vector<char> data(static_cast<size_t>(length));
    file.seekg(0);
    if (!file.read(data.data(), static_cast<std::streamsize>(data.size()))) throw std::runtime_error("读取模型文件不完整: " + path);
    return data;
}

/// 读取 protobuf 的变长无符号整数，检查截断及溢出，避免依赖额外 protobuf 运行库。
uint64_t varint(std::string_view data, size_t& offset) {
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64 && offset < data.size(); shift += 7) {
        auto byte = static_cast<unsigned char>(data[offset++]);
        if (shift == 63 && byte > 1) throw std::runtime_error("ONNX protobuf 整数溢出");
        value |= static_cast<uint64_t>(byte & 127) << shift;
        if (!(byte & 128)) return value;
    }
    throw std::runtime_error("ONNX protobuf 整数被截断");
}

/// 跳过不关心的 protobuf 字段，仅对长度字段返回有效数据片段。
std::string_view field(std::string_view data, size_t& offset, unsigned wire) {
    size_t length = 0;
    if (wire == 0) { varint(data, offset); return {}; }
    if (wire == 1) length = 8;
    else if (wire == 5) length = 4;
    else if (wire == 2) {
        const auto count = varint(data, offset);
        if (count > data.size() - offset) throw std::runtime_error("ONNX protobuf 字段被截断");
        length = static_cast<size_t>(count);
    } else throw std::runtime_error("ONNX protobuf 字段类型不受支持");
    if (length > data.size() - offset) throw std::runtime_error("ONNX protobuf 字段越界");
    auto result = data.substr(offset, length);
    offset += length;
    return wire == 2 ? result : std::string_view{};
}

/// 仅提取 ONNX ModelProto.metadata_props（字段 14），网络解析仍完全交给 NVIDIA 解析器。
std::map<std::string, std::string> readMetadata(const std::vector<char>& bytes) {
    std::map<std::string, std::string> metadata;
    std::string_view data(bytes.data(), bytes.size());
    size_t offset = 0;
    while (offset < data.size()) {
        auto tag = varint(data, offset);
        auto entry = field(data, offset, static_cast<unsigned>(tag & 7));
        if ((tag >> 3) != 14 || (tag & 7) != 2) continue;
        std::string key, value;
        size_t cursor = 0;
        while (cursor < entry.size()) {
            auto itemTag = varint(entry, cursor);
            auto item = field(entry, cursor, static_cast<unsigned>(itemTag & 7));
            if ((itemTag & 7) != 2) continue;
            if ((itemTag >> 3) == 1) key.assign(item);
            if ((itemTag >> 3) == 2) value.assign(item);
        }
        if (!key.empty()) metadata[key] = value;
    }
    return metadata;
}

/// 用十六进制编码元数据，避免网络名中的分隔符与任意 UTF-8 文本发生冲突。
std::string hex(const std::string& value) {
    static const char alphabet[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2);
    for (unsigned char c : value) { result += alphabet[c >> 4]; result += alphabet[c & 15]; }
    return result;
}

/// 解码本工程写入的网络名片段，遇到损坏字符时报错。
std::string unhex(const std::string& value) {
    if (value.size() % 2) throw std::runtime_error("引擎元数据长度无效");
    std::string result;
    for (size_t i = 0; i < value.size(); i += 2) {
        unsigned int byte = 0;
        auto pair = value.substr(i, 2);
        if (pair.find_first_not_of("0123456789abcdef") != std::string::npos) throw std::runtime_error("引擎元数据编码损坏");
        byte = static_cast<unsigned>(std::stoul(pair, nullptr, 16));
        result += static_cast<char>(byte);
    }
    return result;
}

/// 将 ONNX 元数据编码为网络名，序列化之后可从原生 engine 独立恢复。
std::string encodeMetadata(const std::map<std::string, std::string>& metadata) {
    std::string name = "trt_model_v1";
    for (const auto& pair : metadata) name += ";" + hex(pair.first) + "=" + hex(pair.second);
    return name;
}

/// 从引擎网络名恢复元数据；兼容外部生成但不携带元数据的普通引擎。
std::map<std::string, std::string> decodeMetadata(const char* name) {
    std::map<std::string, std::string> metadata;
    if (!name || std::string(name).rfind("trt_model_v1", 0) != 0) return metadata;
    std::istringstream stream(std::string(name).substr(12));
    std::string item;
    while (std::getline(stream, item, ';')) {
        if (item.empty()) continue;
        auto separator = item.find('=');
        if (separator == std::string::npos) throw std::runtime_error("引擎元数据字段损坏");
        metadata[unhex(item.substr(0, separator))] = unhex(item.substr(separator + 1));
    }
    return metadata;
}

/// 将维度转为普通整数数组，供模型层查询。
std::vector<int> shapeOf(nvinfer1::Dims dims) {
    return std::vector<int>(dims.d, dims.d + dims.nbDims);
}

/// 计算静态张量元素数，拒绝动态、空维度及乘法溢出。
size_t volume(const std::vector<int>& shape) {
    size_t result = 1;
    for (int dim : shape) {
        if (dim <= 0 || result > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim))
            throw std::runtime_error("张量包含未解析的动态维度或元素数溢出");
        result *= static_cast<size_t>(dim);
    }
    return result;
}

/// 返回 TensorRT 类型的实际字节宽度，输出统一在取回后转换为 float。
size_t typeSize(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: case nvinfer1::DataType::kBOOL: return 1;
    default: throw std::runtime_error("引擎输出类型不受支持");
    }
}

/// 通过 IEEE754 位表示将 FP16 转成 FP32，兼容无原生半精度指令的 CPU。
float halfToFloat(uint16_t bits) {
    const int sign = (bits & 0x8000) ? -1 : 1;
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    if (!exponent) return sign * std::ldexp(static_cast<float>(fraction), -24);
    if (exponent == 31) return fraction ? std::numeric_limits<float>::quiet_NaN() : sign * std::numeric_limits<float>::infinity();
    return sign * std::ldexp(1.0f + static_cast<float>(fraction) / 1024.0f, exponent - 15);
}

/// 将不同 TensorRT 输出类型转换为 float 数组，布尔/整型输出保留其数值语义。
void convertOutput(const std::vector<unsigned char>& source, nvinfer1::DataType type, std::vector<float>& result) {
    if (type == nvinfer1::DataType::kFLOAT) { std::memcpy(result.data(), source.data(), result.size() * sizeof(float)); return; }
    for (size_t i = 0; i < result.size(); ++i) {
        if (type == nvinfer1::DataType::kHALF) { uint16_t v; std::memcpy(&v, source.data() + i * 2, 2); result[i] = halfToFloat(v); }
        else if (type == nvinfer1::DataType::kINT32) { int32_t v; std::memcpy(&v, source.data() + i * 4, 4); result[i] = static_cast<float>(v); }
        else if (type == nvinfer1::DataType::kINT8) result[i] = static_cast<float>(static_cast<int8_t>(source[i]));
        else result[i] = source[i] ? 1.0f : 0.0f;
    }
}

/// 从 ONNX names 字典的数值键或独立类别数字段恢复类别数，不根据输出结果猜类别。
int classCount(const std::map<std::string, std::string>& metadata) {
    for (const char* key : {"nc", "class_count", "anomalib.class_count"}) {
        auto found = metadata.find(key);
        if (found != metadata.end()) {
            const auto count = std::stoll(found->second);
            if (count <= 0 || count > std::numeric_limits<int>::max()) throw std::runtime_error("ONNX 类别数无效");
            return static_cast<int>(count);
        }
    }
    auto names = metadata.find("names");
    if (names == metadata.end()) return 0;
    const std::regex key("(?:^|[,\\{])\\s*['\"]?([0-9]+)['\"]?\\s*:");
    int count = 0, maxIndex = -1;
    for (auto it = std::sregex_iterator(names->second.begin(), names->second.end(), key); it != std::sregex_iterator(); ++it) {
        maxIndex = std::max(maxIndex, std::stoi((*it)[1])); ++count;
    }
    if (count && maxIndex + 1 != count) throw std::runtime_error("ONNX names 类别编号必须从 0 连续排列");
    return count;
}

/// 仅保存模型运行所需的元数据，并检查 TensorRT 网络名称 4096 字节的硬限制。
std::string runtimeMetadataName(const std::map<std::string, std::string>& source) {
    std::map<std::string, std::string> compact;
    const int count = classCount(source);
    if (count > 0) compact["class_count"] = std::to_string(count);
    for (const char* key : {"batch", "imgsz", "task", "model_family", "end2end"}) {
        const auto found = source.find(key);
        if (found != source.end()) compact[key] = found->second;
    }
    // args 可能包含冗长的导出参数，只提取真正影响输出解码的 end2end 布尔值。
    if (!compact.count("end2end")) {
        const auto args = source.find("args");
        if (args != source.end()) {
            std::smatch match;
            if (std::regex_search(args->second, match, std::regex("['\"]?end2end['\"]?\\s*:\\s*(True|true|False|false|0|1)")))
                compact["end2end"] = (match[1] == "True" || match[1] == "true" || match[1] == "1") ? "true" : "false";
        }
    }
    for (const auto& item : source) {
        if (item.first.rfind("anomalib.", 0) == 0 && item.first != "anomalib.source_checkpoint"
            && item.first.rfind("anomalib.raw_", 0) != 0) compact[item.first] = item.second;
        if (item.first.rfind("trt_model.", 0) == 0) compact[item.first] = item.second;
    }
    const auto encoded = encodeMetadata(compact);
    if (encoded.size() >= 4096) throw std::runtime_error("模型运行元数据超过 TensorRT 网络名称 4096 字节限制，请缩短类别名称或异常模型说明");
    return encoded;
}

/// 使用模型保存的训练尺寸解析动态 NCHW；缺失尺寸时要求重新导出静态 ONNX。
nvinfer1::Dims resolveInput(nvinfer1::Dims dims, const std::map<std::string, std::string>& metadata) {
    if (dims.nbDims != 4) throw std::runtime_error("只支持单个 NCHW 图像输入");
    auto integer = [&](const char* key, int fallback) {
        const auto found = metadata.find(key);
        if (found == metadata.end()) return fallback;
        const auto value = std::stoll(found->second);
        if (value <= 0 || value > std::numeric_limits<int>::max()) throw std::runtime_error("ONNX 输入尺寸元数据无效");
        return static_cast<int>(value);
    };
    if (dims.d[0] < 0) dims.d[0] = integer("batch", 1);
    if (dims.d[1] < 0) dims.d[1] = 3;
    int height = integer("anomalib.resize_height", 0), width = integer("anomalib.resize_width", 0);
    const auto imageSize = metadata.find("imgsz");
    if (imageSize != metadata.end()) {
        const std::regex number("[0-9]+");
        std::vector<int> sizes;
        for (auto it = std::sregex_iterator(imageSize->second.begin(), imageSize->second.end(), number); it != std::sregex_iterator(); ++it)
            sizes.push_back(std::stoi(it->str()));
        if (sizes.size() == 2) { height = sizes[0]; width = sizes[1]; }
    }
    if (dims.d[2] < 0) dims.d[2] = height;
    if (dims.d[3] < 0) dims.d[3] = width;
    volume(shapeOf(dims));
    return dims;
}


}

struct Engine::Impl {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    int device = 0, inputIndex = -1;
    ModelInfo info;
    std::vector<void*> buffers;
    std::vector<std::vector<unsigned char>> outputBytes;
    std::vector<int> outputIndices;
    /// 先同步流再释放缓冲区，释放失败记录 CUDA 错误，析构不传播异常。
    ~Impl() {
        int previous = 0;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        context.reset();
        for (void* buffer : buffers) if (buffer) cudaFree(buffer);
        if (stream) cudaStreamDestroy(stream);
        engine.reset(); runtime.reset();
        cudaSetDevice(previous);
    }
};

/// 创建空模型，加载时才申请 CUDA 资源。
Engine::Engine() = default;
/// 由内部资源对象按依赖顺序释放 TensorRT 和 CUDA 资源。
Engine::~Engine() = default;

/// ONNX 自动构建同目录 FP32 缓存，再加载并验证单图像输入引擎和独立 GPU 缓冲区。
void Engine::load(const std::string& path) {
    impl_.reset();
    auto extension = std::filesystem::u8path(path).extension().u8string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (extension == ".onnx") {
        const auto source = std::filesystem::u8path(path);
        auto cached = source;
        cached.replace_extension(".trt8616.fp32.engine");
        if (!std::filesystem::exists(cached) || std::filesystem::last_write_time(cached) < std::filesystem::last_write_time(source))
            buildEngine(path, cached.u8string(), false);
        load(cached.u8string());
        return;
    }
    auto next = std::make_unique<Impl>();
    checkCuda(cudaGetDevice(&next->device), "读取 CUDA 设备失败");
    initializePlugins();
    auto data = readBinary(path);
    next->runtime.reset(nvinfer1::createInferRuntime(next->logger));
    if (!next->runtime) throw std::runtime_error("创建 TensorRT Runtime 失败: " + next->logger.errors());
    next->engine.reset(next->runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!next->engine) throw std::runtime_error("读取 TensorRT 8.6.1.6 引擎失败，请用当前版本重新转换: " + next->logger.errors());
    next->info.metadata = decodeMetadata(next->engine->getName());
    next->info.classes = classCount(next->info.metadata);
    auto args = next->info.metadata.find("args");
    if (args != next->info.metadata.end()) next->info.endToEnd = std::regex_search(args->second, std::regex("['\"]?end2end['\"]?\\s*:\\s*(True|true|1)"));
    const auto endToEnd = next->info.metadata.find("end2end");
    if (endToEnd != next->info.metadata.end()) next->info.endToEnd = endToEnd->second == "true" || endToEnd->second == "True" || endToEnd->second == "1";
    next->context.reset(next->engine->createExecutionContext());
    if (!next->context) throw std::runtime_error("创建 TensorRT 执行上下文失败: " + next->logger.errors());
    const int bindings = next->engine->getNbBindings();
    if (next->engine->getNbOptimizationProfiles() > 1) throw std::runtime_error("仅支持一个优化配置的图像引擎");
    next->buffers.resize(bindings, nullptr);
    for (int i = 0; i < bindings; ++i) {
        if (next->engine->getBindingFormat(i) != nvinfer1::TensorFormat::kLINEAR)
            throw std::runtime_error("仅支持线性 NCHW 输入输出绑定");
        if (!next->engine->bindingIsInput(i)) continue;
        if (next->inputIndex >= 0 || next->engine->isShapeBinding(i)) throw std::runtime_error("仅支持单图像输入，不支持额外形状输入");
        next->inputIndex = i;
        auto dims = next->engine->getBindingDimensions(i);
        if (std::find(dims.d, dims.d + dims.nbDims, -1) != dims.d + dims.nbDims) {
            dims = next->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kOPT);
            if (!next->context->setBindingDimensions(i, dims)) throw std::runtime_error("设置引擎输入维度失败");
        }
        if (dims.nbDims != 4 || dims.d[1] != 3 || next->engine->getBindingDataType(i) != nvinfer1::DataType::kFLOAT)
            throw std::runtime_error("图像输入必须是 NCHW、3 通道、FP32");
        next->info.batch = dims.d[0]; next->info.channels = dims.d[1];
        next->info.height = dims.d[2]; next->info.width = dims.d[3];
        next->info.inputName = next->engine->getBindingName(i);
    }
    if (next->inputIndex < 0 || !next->context->allInputDimensionsSpecified()) throw std::runtime_error("引擎输入缺失或动态维度未指定");
    for (int i = 0; i < bindings; ++i) {
        auto shape = shapeOf(next->context->getBindingDimensions(i));
        const size_t count = volume(shape);
        const size_t unit = typeSize(next->engine->getBindingDataType(i));
        if (count > std::numeric_limits<size_t>::max() / unit) throw std::runtime_error("模型缓冲区字节数溢出");
        checkCuda(cudaMalloc(&next->buffers[i], count * unit), "申请模型 GPU 缓冲区失败");
        if (i != next->inputIndex) {
            next->info.outputs.push_back({next->engine->getBindingName(i), std::move(shape), {}});
            next->outputIndices.push_back(i);
            next->outputBytes.emplace_back(count * unit);
        }
    }
    if (next->outputIndices.empty()) throw std::runtime_error("引擎没有输出");
    checkCuda(cudaStreamCreate(&next->stream), "创建 CUDA 流失败");
    impl_ = std::move(next);
}

/// 返回模型信息，拒绝在初始化前读取。
const ModelInfo& Engine::info() const {
    if (!impl_) throw std::runtime_error("模型尚未初始化");
    return impl_->info;
}

/// 复用绑定缓冲区执行当前 batch，完成设备同步后再转换并返回输出。
std::vector<Tensor> Engine::infer(const std::vector<float>& input) {
    if (!impl_) throw std::runtime_error("模型尚未初始化");
    auto& state = *impl_;
    DeviceGuard device(state.device);
    if (input.size() != volume({state.info.batch, state.info.channels, state.info.height, state.info.width}))
        throw std::runtime_error("输入数据量与模型 NCHW 不匹配");
    try {
        checkCuda(cudaMemcpyAsync(state.buffers[state.inputIndex], input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, state.stream), "复制模型输入失败");
        if (!state.context->enqueueV2(state.buffers.data(), state.stream, nullptr)) throw std::runtime_error("TensorRT 推理失败: " + state.logger.errors());
        for (size_t i = 0; i < state.outputIndices.size(); ++i)
            checkCuda(cudaMemcpyAsync(state.outputBytes[i].data(), state.buffers[state.outputIndices[i]], state.outputBytes[i].size(), cudaMemcpyDeviceToHost, state.stream), "复制模型输出失败");
        checkCuda(cudaStreamSynchronize(state.stream), "等待 TensorRT 推理失败");
    } catch (...) {
        // 保证异步传输不再访问输入数据，调用方才可安全释放本次输入。
        cudaStreamSynchronize(state.stream);
        throw;
    }
    auto outputs = state.info.outputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i].values.resize(volume(outputs[i].shape));
        convertOutput(state.outputBytes[i], state.engine->getBindingDataType(state.outputIndices[i]), outputs[i].values);
    }
    return outputs;
}

/// 解析 ONNX 后直接构建引擎，动态尺寸来自元数据；Dinomaly 固定 FP32 以保持重建与阈值判定精度。
void buildEngine(const std::string& onnx, const std::string& engine, bool fp16) {
    Logger logger;
    if (onnx.empty() || engine.empty()) throw std::invalid_argument("ONNX 和引擎输出路径不能为空");
    const auto source = std::filesystem::absolute(std::filesystem::u8path(onnx)).lexically_normal();
    const auto destination = std::filesystem::absolute(std::filesystem::u8path(engine)).lexically_normal();
    if (source == destination) throw std::runtime_error("ONNX 和引擎输出路径不能相同");
    auto bytes = readBinary(onnx);
    auto metadata = readMetadata(bytes);
    prepareAnomalyOnnx(bytes, metadata);
    initializePlugins();
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("创建 TensorRT Builder 失败: " + logger.errors());
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(1U << static_cast<unsigned>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if (!network) throw std::runtime_error("创建 ONNX 网络失败");
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) throw std::runtime_error("创建 ONNX 解析器失败");
    // 由本工程按 UTF-8 路径读取，再从内存解析；model_path 保留外部权重的相对路径基准。
    const auto modelPath = source.u8string();
    const bool parsed = parser->parse(bytes.data(), bytes.size(), modelPath.c_str());
    if (!parsed) {
        std::string details;
        for (int i = 0; i < parser->getNbErrors(); ++i) details += std::string(parser->getError(i)->desc()) + "\n";
        throw std::runtime_error("TensorRT 8.6 ONNX 解析失败: " + details + logger.errors());
    }
    bytes.clear(); bytes.shrink_to_fit();
    if (network->getNbInputs() != 1) throw std::runtime_error("ONNX 必须只有一个图像输入");
    auto* input = network->getInput(0);
    if (input->getType() != nvinfer1::DataType::kFLOAT) throw std::runtime_error("请导出 FP32 输入的 ONNX，FP16 由构建选项选择");
    auto dims = input->getDimensions();
    const auto resolved = resolveInput(dims, metadata);
    if (resolved.d[1] != 3) throw std::runtime_error("ONNX 必须使用 3 通道 NCHW 图像输入");
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) throw std::runtime_error("创建 TensorRT 构建配置失败");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    // 异常分数对数值精度敏感；FP32 模式也关闭默认 TF32，便于和 ONNX 参考值对照。
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    const auto modelType = metadata.find("anomalib.model_type");
    const bool dinomaly = modelType != metadata.end() && modelType->second == "Dinomaly";
    const bool useFp16 = fp16 && !dinomaly && builder->platformHasFastFp16();
    if (useFp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
    metadata["trt_model.requested_precision"] = fp16 ? "fp16" : "fp32";
    metadata["trt_model.effective_precision"] = useFp16 ? "fp16_allowed" : "fp32";
    if (dinomaly) {
        // 线性注意力累加超过65504；仅保护该段仍会使0.5011的缺陷得分跌至0.49785。
        // 保留训练时的FP32计算和原阈值，避免用调低阈值掩盖精度损失。
        metadata["trt_model.precision_policy"] = "dinomaly_fp32_required_v1";
        if (fp16) logger.log(nvinfer1::ILogger::Severity::kWARNING,
            "Dinomaly的FP16存在注意力溢出和临界样本漏检，已按FP32精度保护构建；其他模型不受影响。");
    }
    // IOptimizationProfile 由 builder 管理，其析构函数受保护，不能交给 unique_ptr 删除。
    nvinfer1::IOptimizationProfile* profile = nullptr;
    if (std::find(dims.d, dims.d + dims.nbDims, -1) != dims.d + dims.nbDims) {
        profile = builder->createOptimizationProfile();
        if (!profile || !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, resolved)
            || !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, resolved)
            || !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, resolved)
            || config->addOptimizationProfile(profile) < 0) throw std::runtime_error("无法根据 ONNX 元数据设置动态输入尺寸");
    }
    const std::string networkName = runtimeMetadataName(metadata);
    network->setName(networkName.c_str());
    std::unique_ptr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) throw std::runtime_error("构建 TensorRT 引擎失败: " + logger.errors());
    // 先写同目录临时文件，全部成功后才替换目标，防止失败留下半个引擎文件。
    auto temporary = destination;
    temporary += ".building." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
        + "." + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    try {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file || !file.write(static_cast<const char*>(serialized->data()), static_cast<std::streamsize>(serialized->size())) || !file.flush())
            throw std::runtime_error("保存引擎失败: " + engine);
        file.close();
#ifdef _WIN32
        // Windows rename 不覆盖已有文件，先用系统原子替换以保护已经存在的 engine。
        if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("替换引擎文件失败，系统错误码: " + std::to_string(GetLastError()));
#else
        std::filesystem::rename(temporary, destination);
#endif
    } catch (...) { std::error_code ignored; std::filesystem::remove(temporary, ignored); throw; }
}
}
