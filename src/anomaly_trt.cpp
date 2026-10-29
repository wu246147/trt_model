#include "anomaly_trt.h"
#include "anomaly_preprocess.h"
#include "trt_engine.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace {
/// 验证部署约定，失败时带明确原因抛出异常供外层转换返回码。
void require(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }

/// 从枚举取得模型元数据中约定的名称，拒绝无效模型编号。
std::string kindName(AnomalyKind kind) {
    switch (kind) {
    case AnomalyKind::AnomalyDINO: return "AnomalyDINO";
    case AnomalyKind::Dinomaly: return "Dinomaly";
    case AnomalyKind::EfficientAD: return "EfficientAD";
    }
    throw std::invalid_argument("异常检测模型编号无效");
}

/// 读取必需元数据，禁止用固定默认值掩盖校准或输入协议缺失。
std::string metadataValue(const trt_model::ModelInfo& model, const std::string& key) {
    const auto found = model.metadata.find(key);
    require(found != model.metadata.end() && !found->second.empty(), "异常模型缺少元数据: " + key);
    return found->second;
}

/// 读取正整数元数据，验证完整转换及int边界。
int positiveMetadata(const trt_model::ModelInfo& model, const std::string& key) {
    const auto text = metadataValue(model, key); size_t end = 0;
    const auto value = std::stoll(text, &end);
    require(end == text.size() && value > 0 && value <= std::numeric_limits<int>::max(), "异常模型尺寸元数据无效: " + key);
    return static_cast<int>(value);
}

/// 读取校准后的归一化阈值，必须为[0,1]中的有限数。
float thresholdMetadata(const trt_model::ModelInfo& model, const std::string& key) {
    const auto text = metadataValue(model, key); size_t end = 0;
    const auto value = std::stof(text, &end);
    require(end == text.size() && std::isfinite(value) && value >= 0 && value <= 1, "异常模型阈值元数据无效: " + key);
    return value;
}

/// 将支持的8位灰度/BGR/BGRA输入转换为BGR，BGR输入通过引用头复用。
cv::Mat asBgr(const cv::Mat& frame) {
    require(!frame.empty() && frame.dims == 2 && frame.depth() == CV_8U, "请输入非空的8位灰度、BGR或BGRA图像");
    cv::Mat bgr;
    switch (frame.channels()) {
    case 1: cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR); break;
    case 3: bgr = frame; break;
    case 4: cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR); break;
    default: throw std::runtime_error("异常检测仅支持1、3或4通道图像");
    }
    return bgr;
}

/// 按名称读取TensorRT输出，不依赖导出器输出顺序。
const trt_model::Tensor& outputNamed(const std::vector<trt_model::Tensor>& outputs, const std::string& name) {
    for (const auto& output : outputs) if (output.name == name) return output;
    throw std::runtime_error("异常模型缺少输出: " + name);
}

/// 克隆所有结果矩阵，防止调用方修改模型内部保存的结果。
AnomalyResult cloneResult(const AnomalyResult& source) {
    AnomalyResult result = source;
    result.anomalyMap = source.anomalyMap.clone(); result.mask = source.mask.clone();
    result.heatmap = source.heatmap.clone(); result.overlay = source.overlay.clone(); return result;
}
}

struct AnomalyTRT::Impl {
    /// 保存期望模型名称，初始化之前不分配GPU资源。
    explicit Impl(AnomalyKind kind) : expected(kindName(kind)) {}
    mutable std::mutex mutex;
    std::string expected, error;
    std::unique_ptr<trt_model::Engine> engine;
    AnomalyModelInfo info;
    AnomalyResult prediction;
    double imageOverride = -1, pixelOverride = -1;
    float defaultImageThreshold = 0.5f;
    bool bicubic = false, antialias = true;
    int cutSize = 640, overlap = 128;
};

/// 创建特定异常模型执行器，验证模型枚举。
AnomalyTRT::AnomalyTRT(AnomalyKind kind) : impl_(new Impl(kind)) {}
/// 自动释放本实例持有的TensorRT及结果资源。
AnomalyTRT::~AnomalyTRT() = default;
/// 创建AnomalyDINO类型的执行器。
AnomalyDINOTRT::AnomalyDINOTRT() : AnomalyTRT(AnomalyKind::AnomalyDINO) {}
/// 创建Dinomaly类型的执行器。
DinomalyTRT::DinomalyTRT() : AnomalyTRT(AnomalyKind::Dinomaly) {}
/// 创建EfficientAD类型的执行器。
EfficientADTRT::EfficientADTRT() : AnomalyTRT(AnomalyKind::EfficientAD) {}

/// 调用公共ONNX解析器构建原生引擎，保存全部校准信息，失败保留详细错误。
int AnomalyTRT::onnxToEngine(std::string onnxPath, std::string enginePath, bool fp16) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->error.clear();
    try { trt_model::buildEngine(onnxPath, enginePath, fp16); return 0; }
    catch (const std::exception& error) { impl_->error = "异常模型转换失败: " + std::string(error.what()); return -1; }
}

/// 加载本类型的校准引擎并从元数据读取输入/输出约定，任何失败均释放旧模型。
int AnomalyTRT::initModel(std::string enginePath) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->engine.reset(); impl_->prediction = {}; impl_->info = {}; impl_->error.clear();
    try {
        auto engine = std::make_unique<trt_model::Engine>(); engine->load(enginePath);
        const auto& model = engine->info();
        require(metadataValue(model, "anomalib.contract_version") == "1", "异常模型合同版本不受支持");
        require(metadataValue(model, "anomalib.model_type") == impl_->expected, "模型类型不匹配，当前接口需要 " + impl_->expected);
        require(metadataValue(model, "anomalib.input_layout") == "NCHW" && metadataValue(model, "anomalib.input_color") == "RGB" && metadataValue(model, "anomalib.input_range") == "0,1", "异常模型输入协议不受支持");
        require(metadataValue(model, "anomalib.preprocessing") == "external_antialias" && metadataValue(model, "anomalib.postprocessing") == "embedded", "请使用本工程直接从已校准ONNX构建异常引擎");
        require(model.batch == 1 && model.channels == 3 && positiveMetadata(model, "anomalib.class_count") == 2, "异常模型要求单批次RGB输入和正常/异常两类输出");
        AnomalyModelInfo info;
        info.modelType = impl_->expected; info.normalClass = metadataValue(model, "anomalib.class_0"); info.anomalyClass = metadataValue(model, "anomalib.class_1");
        info.resizeSize = cv::Size(positiveMetadata(model, "anomalib.resize_width"), positiveMetadata(model, "anomalib.resize_height"));
        info.networkSize = cv::Size(positiveMetadata(model, "anomalib.network_width"), positiveMetadata(model, "anomalib.network_height"));
        require(info.resizeSize == cv::Size(model.width, model.height), "异常引擎输入尺寸与模型元数据不匹配");
        info.batchSize = model.batch; info.preprocessingEmbedded = false;
        impl_->defaultImageThreshold = thresholdMetadata(model, "anomalib.image_threshold");
        info.defaultPixelThreshold = thresholdMetadata(model, "anomalib.pixel_threshold");
        info.imageThreshold = impl_->imageOverride < 0 ? impl_->defaultImageThreshold : static_cast<float>(impl_->imageOverride);
        info.pixelThreshold = impl_->pixelOverride < 0 ? info.defaultPixelThreshold : static_cast<float>(impl_->pixelOverride);
        const auto interpolation = metadataValue(model, "anomalib.resize_interpolation");
        require(interpolation == "bilinear" || interpolation == "bicubic", "异常模型缩放插值方式不受支持");
        impl_->bicubic = interpolation == "bicubic";
        const auto antialias = metadataValue(model, "anomalib.resize_antialias");
        require(antialias == "true", "当前异常部署合同要求训练缩放采用抗锯齿插值");
        impl_->antialias = true;
        const auto& score = outputNamed(model.outputs, "pred_score"); const auto& map = outputNamed(model.outputs, "anomaly_map");
        bool singleScore = true; for (int extent : score.shape) singleScore = singleScore && extent == 1;
        // TRT可能把ONNX标量规范化为[1,1]；所有轴均为1时仍表示同一个图片得分。
        require(singleScore && map.shape == std::vector<int>({1, 1, info.networkSize.height, info.networkSize.width}), "异常模型输出形状不符合校准合同");
        impl_->info = std::move(info); impl_->engine = std::move(engine); return 0;
    } catch (const std::exception& error) { impl_->error = "异常模型初始化失败: " + std::string(error.what()); return -1; }
}

/// 设置独立图片与像素阈值；-1恢复各模型保存的默认校准阈值。
int AnomalyTRT::setPara(double imageThreshold, double pixelThreshold) {
    std::lock_guard<std::mutex> lock(impl_->mutex); impl_->error.clear();
    const auto valid = [](double value) { return std::isfinite(value) && (value == -1 || (value >= 0 && value <= 1)); };
    if (!valid(imageThreshold) || !valid(pixelThreshold)) { impl_->error = "图片和像素阈值必须为[0,1]中的有限数，或用-1恢复模型默认值"; return -2; }
    impl_->imageOverride = imageThreshold; impl_->pixelOverride = pixelThreshold;
    impl_->info.imageThreshold = imageThreshold < 0 ? impl_->defaultImageThreshold : static_cast<float>(imageThreshold);
    impl_->info.pixelThreshold = pixelThreshold < 0 ? impl_->info.defaultPixelThreshold : static_cast<float>(pixelThreshold);
    return 0;
}

/// 校验并设置切片几何参数，失败不覆盖有效配置，也不修改模型校准阈值。
int AnomalyTRT::setCutPara(int cutSize, int overlap) {
    std::lock_guard<std::mutex> lock(impl_->mutex); impl_->error.clear();
    if (cutSize <= 0 || overlap < 0 || overlap >= cutSize) { impl_->error = "切片尺寸必须大于0，重叠像素必须满足0<=overlap<cutSize"; return -2; }
    impl_->cutSize = cutSize; impl_->overlap = overlap; return 0;
}

/// 原子校验并设置YOLO风格切片与像素阈值参数，保留已设定的图片阈值。
int AnomalyTRT::setPara(int cutSize, int overlappingSize, double pixelThreshold) {
    std::lock_guard<std::mutex> lock(impl_->mutex); impl_->error.clear();
    if (cutSize <= 0 || overlappingSize < 0 || overlappingSize >= cutSize || !std::isfinite(pixelThreshold) ||
        !(pixelThreshold == -1 || (pixelThreshold >= 0 && pixelThreshold <= 1))) {
        impl_->error = "切片尺寸、重叠像素或像素阈值无效"; return -2;
    }
    impl_->cutSize = cutSize; impl_->overlap = overlappingSize; impl_->pixelOverride = pixelThreshold;
    impl_->info.pixelThreshold = pixelThreshold < 0 ? impl_->info.defaultPixelThreshold : static_cast<float>(pixelThreshold); return 0;
}

/// 对整图执行一次模型推理，len成功为1，失败为0。
int AnomalyTRT::runModel(cv::Mat& inputImage, int& len) { return run(inputImage, len, false); }

/// 对所有覆盖切片顺序复用同一个GPU上下文并融合整图结果，len成功为1。
int AnomalyTRT::runCutModel(cv::Mat& inputImage, int& len) { return run(inputImage, len, true); }

/// 提供与YOLO一致的切图命名，不额外获取锁以避免递归加锁。
int AnomalyTRT::runModelCutImage(cv::Mat& inputImage, int& len) { return run(inputImage, len, true); }

/// 执行训练等价缩放、整图或切片GPU推理及统一可视化；融合中只保留整图画布和当前切片。
int AnomalyTRT::run(cv::Mat& inputImage, int& len, bool cuts) {
    std::lock_guard<std::mutex> lock(impl_->mutex); len = 0; impl_->prediction = {}; impl_->error.clear();
    if (!impl_->engine) { impl_->error = "异常模型尚未初始化"; return -3; }
    try {
        const auto start = std::chrono::steady_clock::now(); const auto bgr = asBgr(inputImage);
        AnomalyResult prediction;
        prediction.anomalyMap = cv::Mat::zeros(bgr.size(), CV_32FC1); prediction.mask = cv::Mat::zeros(bgr.size(), CV_8UC1);
        const int tileSize = cuts ? impl_->cutSize : std::max(bgr.rows, bgr.cols);
        const int step = cuts ? impl_->cutSize - impl_->overlap : tileSize;
        for (int y = 0;;) {
            const int height = std::min(tileSize, bgr.rows - y);
            for (int x = 0;;) {
                const int width = std::min(tileSize, bgr.cols - x); const cv::Rect region(x, y, width, height);
                const auto input = trt_model::anomalyInput(bgr(region), impl_->info.resizeSize, impl_->bicubic, impl_->antialias);
                const auto outputs = impl_->engine->infer(input);
                const auto& score = outputNamed(outputs, "pred_score"); const auto& map = outputNamed(outputs, "anomaly_map");
                const size_t mapElements = static_cast<size_t>(impl_->info.networkSize.width) * impl_->info.networkSize.height;
                require(score.values.size() == 1 && map.values.size() == mapElements, "异常模型输出元素数量不匹配");
                require(std::isfinite(score.values[0]) && score.values[0] >= 0 && score.values[0] <= 1, "异常模型返回无效的归一化图片得分");
                prediction.score = std::max(prediction.score, score.values[0]);
                cv::Mat networkMap(impl_->info.networkSize, CV_32FC1, const_cast<float*>(map.values.data()));
                require(cv::checkRange(networkMap, true, nullptr, 0.0, 1.000001), "异常模型返回无效的归一化异常图");
                cv::Mat tileMap, tileMask;
                cv::resize(networkMap, tileMap, region.size(), 0, 0, cv::INTER_LINEAR);
                // 先在模型分辨率阈值化再最近邻恢复；重叠区域采用OR，避免不同分辨率插值改变阈值含义。
                const cv::Mat networkMask = networkMap > impl_->info.pixelThreshold;
                cv::resize(networkMask, tileMask, region.size(), 0, 0, cv::INTER_NEAREST);
                auto mapRegion = prediction.anomalyMap(region); auto maskRegion = prediction.mask(region);
                cv::max(mapRegion, tileMap, mapRegion); cv::bitwise_or(maskRegion, tileMask, maskRegion);
                // 到达右/下边缘立即结束，保证不足整片的边界仍被覆盖且坐标加法不会溢出。
                if (x + width == bgr.cols) break;
                x += step;
            }
            if (y + height == bgr.rows) break;
            y += step;
        }
        prediction.classId = prediction.score > impl_->info.imageThreshold ? 1 : 0;
        prediction.className = prediction.classId ? impl_->info.anomalyClass : impl_->info.normalClass;
        cv::Mat intensity, colors, blended; prediction.anomalyMap.convertTo(intensity, CV_8UC1, 255.0);
        cv::applyColorMap(intensity, colors, cv::COLORMAP_JET);
        prediction.heatmap = cv::Mat::zeros(bgr.size(), CV_8UC3); colors.copyTo(prediction.heatmap, prediction.mask);
        prediction.overlay = bgr.clone(); cv::addWeighted(bgr, 0.6, colors, 0.4, 0, blended); blended.copyTo(prediction.overlay, prediction.mask);
        std::vector<std::vector<cv::Point>> contours; cv::findContours(prediction.mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(prediction.overlay, contours, -1, cv::Scalar(0, 0, 255), 1);
        prediction.elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        impl_->prediction = std::move(prediction); len = 1; return 0;
    } catch (const std::exception& error) { impl_->error = "异常模型推理失败: " + std::string(error.what()); return -1; }
}

/// 深拷贝最近结果，调用方修改结果图像不会影响模型内部状态。
int AnomalyTRT::getResult(AnomalyResult& result) const {
    std::lock_guard<std::mutex> lock(impl_->mutex); result = {}; impl_->error.clear();
    if (impl_->prediction.classId < 0) { impl_->error = "当前没有成功推理的异常检测结果"; return -3; }
    try { result = cloneResult(impl_->prediction); return 0; }
    catch (const std::exception& error) { impl_->error = "复制异常结果失败: " + std::string(error.what()); return -1; }
}

/// 返回模型信息的独立副本，允许初始化前后查询。
AnomalyModelInfo AnomalyTRT::modelInfo() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->info; }
/// 读取最后错误文本，保持与其他实例隔离。
std::string AnomalyTRT::lastError() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->error; }
/// 释放模型及结果并清除错误，可在未初始化时重复调用。
int AnomalyTRT::disposeModel() { std::lock_guard<std::mutex> lock(impl_->mutex); impl_->engine.reset(); impl_->prediction = {}; impl_->info = {}; impl_->error.clear(); return 0; }

/// 创建指定模型类型的独立C句柄；异常不跨越导出边界。
RaivasAnomalyTRTHandle RaivasAnomalyTRT_Create(int kind) { try { return new AnomalyTRT(static_cast<AnomalyKind>(kind)); } catch (...) { return nullptr; } }
/// 销毁C句柄及全部原生资源，空指针可安全传入。
void RaivasAnomalyTRT_Destroy(RaivasAnomalyTRTHandle handle) { delete static_cast<AnomalyTRT*>(handle); }
/// 校验路径后调用对应模型的ONNX转换方法。
int RaivasAnomalyTRT_onnxToEngine(RaivasAnomalyTRTHandle handle, const char* onnxPath, const char* enginePath, bool fp16) {
    if (!handle || !onnxPath || !enginePath) return -2;
    try { return static_cast<AnomalyTRT*>(handle)->onnxToEngine(onnxPath, enginePath, fp16); } catch (...) { return -1; }
}
/// 校验句柄后从engine加载模型。
int RaivasAnomalyTRT_Init(RaivasAnomalyTRTHandle handle, const char* enginePath) {
    if (!handle || !enginePath) return -2;
    try { return static_cast<AnomalyTRT*>(handle)->initModel(enginePath); } catch (...) { return -1; }
}
/// 设置C句柄独立的图片与像素阈值。
int RaivasAnomalyTRT_setPara(RaivasAnomalyTRTHandle handle, double imageThreshold, double pixelThreshold) {
    if (!handle) return -2;
    try { return static_cast<AnomalyTRT*>(handle)->setPara(imageThreshold, pixelThreshold); } catch (...) { return -1; }
}
/// 修改切片几何配置，保持当前校准与用户阈值不变。
int RaivasAnomalyTRT_setCutPara(RaivasAnomalyTRTHandle handle, int cutSize, int overlap) {
    if (!handle) return -2;
    try { return static_cast<AnomalyTRT*>(handle)->setCutPara(cutSize, overlap); } catch (...) { return -1; }
}
/// 通过C句柄推理并深拷贝全部结果，参数无效时清空可用输出。
int RaivasAnomalyTRT_Detect(RaivasAnomalyTRTHandle handle, cv::Mat* frame, AnomalyResult* result) {
    if (!result) return -2;
    try { *result = {}; if (!handle || !frame) return -2; int len = 0; auto* model = static_cast<AnomalyTRT*>(handle); const int status = model->runModel(*frame, len); return status == 0 ? model->getResult(*result) : status; } catch (...) { return -1; }
}
/// 通过C句柄执行切片推理并深拷贝融合结果，异常不会跨越导出边界。
int RaivasAnomalyTRT_DetectCut(RaivasAnomalyTRTHandle handle, cv::Mat* frame, AnomalyResult* result) {
    if (!result) return -2;
    try { *result = {}; if (!handle || !frame) return -2; int len = 0; auto* model = static_cast<AnomalyTRT*>(handle); const int status = model->runCutModel(*frame, len); return status == 0 ? model->getResult(*result) : status; } catch (...) { return -1; }
}
/// 向调用方复制尺寸、类别与有效阈值。
int RaivasAnomalyTRT_GetModelInfo(RaivasAnomalyTRTHandle handle, AnomalyModelInfo* info) {
    if (!handle || !info) return -2;
    try { *info = static_cast<AnomalyTRT*>(handle)->modelInfo(); return 0; } catch (...) { return -1; }
}
/// 复制以零结尾的UTF-8错误文本，允许空缓冲区查询需要的容量。
int RaivasAnomalyTRT_GetLastError(RaivasAnomalyTRTHandle handle, char* buffer, int capacity) {
    if (!handle || capacity < 0 || (!buffer && capacity != 0)) return -2;
    try {
        const auto error = static_cast<AnomalyTRT*>(handle)->lastError();
        if (error.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) return -1;
        if (buffer && capacity > 0) { const size_t count = std::min(error.size(), static_cast<size_t>(capacity - 1)); std::memcpy(buffer, error.data(), count); buffer[count] = '\0'; }
        return static_cast<int>(error.size()) + 1;
    } catch (...) { return -1; }
}
