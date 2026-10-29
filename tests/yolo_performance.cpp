#include "trt_engine.h"
#include "guimodel.h"
#include "yolov5_det.h"
#include "yolo11_det.h"
#include "yolo26_det.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

/// 检查 CUDA 返回码，失败时包含具体操作，禁止将失败计入正常耗时。
void checkCuda(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

/// 保存分位耗时；样本均按毫秒记录，单独给出中位数和P95避免单次调度抖动误导。
struct Statistics { double median = 0, mean = 0, p95 = 0, minimum = 0; };

/// 对非空测量样本排序并计算平均、中位数和P95；输入为空时拒绝生成报告。
Statistics summarize(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("性能测试没有有效样本");
    std::sort(values.begin(), values.end());
    Statistics result;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    result.median = values[values.size() / 2];
    result.p95 = values[std::min(values.size() - 1, static_cast<size_t>(std::ceil(values.size() * 0.95)) - 1)];
    result.minimum = values.front();
    return result;
}

/// 读取模型或图像的二进制内容，路径使用UTF-8；拒绝空文件和不完整读取。
std::vector<char> readBytes(const std::string& path) {
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) throw std::runtime_error("无法读取文件: " + path);
    std::vector<char> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    if (!file.read(bytes.data(), bytes.size())) throw std::runtime_error("文件读取不完整: " + path);
    return bytes;
}

/// 按实际YOLO流程制作RGB/255、114 letterbox的NCHW输入；多余batch槽位清零。
std::vector<float> makeInput(const cv::Mat& image, const trt_model::ModelInfo& info) {
    const float gain = std::min(info.width / static_cast<float>(image.cols), info.height / static_cast<float>(image.rows));
    const int width = static_cast<int>(std::round(image.cols * gain)), height = static_cast<int>(std::round(image.rows * gain));
    const int left = static_cast<int>(std::round((info.width - width) / 2.0 - 0.1));
    const int top = static_cast<int>(std::round((info.height - height) / 2.0 - 0.1));
    cv::Mat resized, padded, rgb, normalized;
    cv::resize(image, resized, {width, height});
    cv::copyMakeBorder(resized, padded, top, info.height - height - top, left, info.width - width - left,
        cv::BORDER_CONSTANT, cv::Scalar::all(114));
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    const size_t plane = static_cast<size_t>(info.width) * info.height;
    std::vector<float> input(plane * info.channels * info.batch, 0);
    std::vector<cv::Mat> planes;
    for (int channel = 0; channel < 3; ++channel) planes.emplace_back(info.height, info.width, CV_32F, input.data() + channel * plane);
    cv::split(normalized, planes);
    return input;
}

/// 执行已预热的CPU墙钟测试；每次fn返回前必须完成对应GPU同步。
template<class Function>
Statistics measure(int iterations, Function fn) {
    for (int i = 0; i < 20; ++i) fn();
    std::vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now(); fn();
        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    return summarize(std::move(samples));
}

/// 纯GPU测试的独立日志对象，生命周期覆盖runtime及engine。
class Logger final : public nvinfer1::ILogger {
public:
    /// 仅输出错误，不在计时区打印性能无关日志。
    void log(Severity severity, const char* text) noexcept override { if (severity <= Severity::kERROR) std::cerr << text << '\n'; }
};

/// GPU独立计时器；输入提前上传，CUDA事件仅包围enqueue，不含预处理、传输和CPU后处理。
class GpuRunner {
public:
    /// 加载单profile静态/固定优化尺寸引擎并创建缓冲区；共享后端须已初始化插件。
    GpuRunner(const std::string& path, const std::vector<float>& input) {
        const auto bytes = readBytes(path);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("创建性能验证runtime失败");
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine_) throw std::runtime_error("性能验证engine反序列化失败");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("性能验证context创建失败");
        buffers_.resize(engine_->getNbBindings(), nullptr);
        for (int i = 0; i < engine_->getNbBindings(); ++i) if (engine_->bindingIsInput(i)) {
            inputIndex_ = i;
            auto dims = engine_->getBindingDimensions(i);
            if (std::find(dims.d, dims.d + dims.nbDims, -1) != dims.d + dims.nbDims)
                if (!context_->setBindingDimensions(i, engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kOPT)))
                    throw std::runtime_error("性能验证输入尺寸设置失败");
        }
        if (inputIndex_ < 0) throw std::runtime_error("性能验证缺少输入");
        // 构造过程的异常由统一清理分支处理，避免部分申请成功后泄漏GPU资源。
        try {
            for (int i = 0; i < engine_->getNbBindings(); ++i) {
                const auto dims = context_->getBindingDimensions(i);
                size_t elements = 1;
                for (int j = 0; j < dims.nbDims; ++j) { if (dims.d[j] <= 0) throw std::runtime_error("性能验证输出动态维度未确定"); elements *= dims.d[j]; }
                const auto type = engine_->getBindingDataType(i);
                const size_t unit = type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kINT32 ? 4 : type == nvinfer1::DataType::kHALF ? 2 : 1;
                checkCuda(cudaMalloc(&buffers_[i], elements * unit), "申请测试GPU缓冲区");
            }
            checkCuda(cudaStreamCreate(&stream_), "创建测试CUDA流");
            checkCuda(cudaEventCreate(&start_), "创建开始事件");
            checkCuda(cudaEventCreate(&stop_), "创建结束事件");
            checkCuda(cudaMemcpyAsync(buffers_[inputIndex_], input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_), "上传测试输入");
            checkCuda(cudaStreamSynchronize(stream_), "等待测试输入");
        } catch (...) { release(); throw; }
    }
    /// 按依赖顺序释放CUDA及TRT资源，不影响其他模型或重置设备。
    ~GpuRunner() { release(); }
    /// 测量多次预热后的GPU执行时间，每次CUDA事件同步后记录结果。
    Statistics benchmark(int iterations) {
        for (int i = 0; i < 30; ++i) enqueue();
        checkCuda(cudaStreamSynchronize(stream_), "等待预热");
        std::vector<double> samples;
        for (int i = 0; i < iterations; ++i) {
            checkCuda(cudaEventRecord(start_, stream_), "记录开始事件"); enqueue();
            checkCuda(cudaEventRecord(stop_, stream_), "记录结束事件");
            checkCuda(cudaEventSynchronize(stop_), "等待结束事件");
            float elapsed = 0; checkCuda(cudaEventElapsedTime(&elapsed, start_, stop_), "读取GPU计时"); samples.push_back(elapsed);
        }
        return summarize(std::move(samples));
    }
private:
    /// 执行已绑定的数据，提交失败立即终止基准测试。
    void enqueue() { if (!context_->enqueueV2(buffers_.data(), stream_, nullptr)) throw std::runtime_error("GPU性能测试enqueue失败"); }
    /// 清理已申请资源，构造失败和析构共用；不抛异常。
    void release() noexcept {
        if (stream_) cudaStreamSynchronize(stream_);
        if (start_) cudaEventDestroy(start_); if (stop_) cudaEventDestroy(stop_);
        start_ = stop_ = nullptr;
        context_.reset();
        for (void*& buffer : buffers_) { if (buffer) cudaFree(buffer); buffer = nullptr; }
        if (stream_) cudaStreamDestroy(stream_); stream_ = nullptr;
        engine_.reset(); runtime_.reset();
    }
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<void*> buffers_;
    int inputIndex_ = -1;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_ = nullptr, stop_ = nullptr;
};

/// 输出单项毫秒统计，字段名固定以供报告脚本读取。
void printStatistics(const char* name, const Statistics& s) {
    std::cout << '"' << name << "\":{\"median_ms\":" << s.median << ",\"mean_ms\":" << s.mean
        << ",\"p95_ms\":" << s.p95 << ",\"min_ms\":" << s.minimum << '}';
}

/// 测量公开DLL整图检测接口，包含实际预处理、传输和NMS，不含原图克隆及绘制。
template<class Detector>
Statistics detectorTime(const std::string& path, cv::Mat& image, int iterations, int& count) {
    Detector detector;
    if (detector.initModel(path) != 0) throw std::runtime_error(detector.getLastError());
    if (detector.setPara(640, 64, 0.5) != 0) throw std::runtime_error(detector.getLastError());
    return measure(iterations, [&] {
        if (detector.runModel(image, count) != 0) throw std::runtime_error(detector.getLastError());
    });
}
}

/// 用指定YOLO检测引擎和真实图片测量GPU、传输封装、检测和界面绘制准备，磁盘读取不计入检测耗时。
int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("用法: yolo_performance version(5/11/26) engine image [iterations=120]");
        const int version = std::stoi(argv[1]), iterations = argc > 4 ? std::stoi(argv[4]) : 120;
        if (iterations < 10 || (version != 5 && version != 11 && version != 26)) throw std::runtime_error("版本或迭代次数无效");
        const auto bytes = readBytes(argv[3]); auto image = cv::imdecode(bytes, cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("图片读取失败");
        auto hostEngine = std::make_unique<trt_model::Engine>(); hostEngine->load(argv[2]);
        const auto info = hostEngine->info(); auto input = makeInput(image, info);
        const auto prep = measure(iterations, [&] { input = makeInput(image, info); });
        Statistics gpu;
        { GpuRunner runner(argv[2], input); gpu = runner.benchmark(iterations); }
        const auto raw = measure(iterations, [&] { auto output = hostEngine->infer(input); });
        hostEngine.reset();
        int apiCount = 0;
        const auto detection = version == 5
            ? detectorTime<RaivasDefectDet5::RaivasDefectDet5>(argv[2], image, iterations, apiCount)
            : version == 11 ? detectorTime<RaivasDefectDet11::RaivasDefectDet11>(argv[2], image, iterations, apiCount)
            : detectorTime<RaivasDefectDet26::RaivasDefectDet26>(argv[2], image, iterations, apiCount);
        const auto task = version == 5 ? GuiTask::Yolo5Det : version == 11 ? GuiTask::Yolo11Det : GuiTask::Yolo26Det;
        auto model = createGuiModel(task); model->load(QString::fromUtf8(argv[2]));
        int count = 0;
        const auto gui = measure(iterations, [&] { auto result = model->run(image, 0.5, false, 640, 64); count = result.count; });
        const auto clone = measure(iterations, [&] { auto copy = image.clone(); });
        if (apiCount != count) throw std::runtime_error("公开DLL与界面适配器的检测数量不一致");
        std::cout << std::setprecision(8) << "{\"version\":" << version << ",\"iterations\":" << iterations
            << ",\"warmup\":20,\"gpu_warmup\":30,\"opencv_threads\":" << cv::getNumThreads()
            << ",\"image_width\":" << image.cols << ",\"image_height\":" << image.rows
            << ",\"input_width\":" << info.width << ",\"input_height\":" << info.height << ",\"batch\":" << info.batch
            << ",\"output_elements\":" << info.outputs[0].shape[1] * info.outputs[0].shape[2] << ",\"detections\":" << count << ',';
        printStatistics("cpu_preprocess", prep); std::cout << ','; printStatistics("gpu_enqueue", gpu); std::cout << ',';
        printStatistics("engine_with_transfers", raw); std::cout << ','; printStatistics("detector_api", detection); std::cout << ',';
        printStatistics("gui_model_run", gui); std::cout << ',';
        printStatistics("original_image_clone", clone); std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
