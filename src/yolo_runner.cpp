#include "yolo_runner.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <regex>
#include <stdexcept>
#include <utility>

namespace trt_model {
namespace {
struct Transform { cv::Mat image; float gain; int left, top, width, height; };
struct Candidate { int classId; float score; cv::Rect2f box; std::vector<float> coefficients; };

/// 按置信度降序排列候选，配合稳定排序保持分数相同目标的模型输出顺序。
bool higherScore(const Candidate& first, const Candidate& second) { return first.score > second.score; }

/// 优先读取明确的 ONNX 分支元数据，返回 1/0 表示启用/关闭，-1 表示未提供，避免用候选数量猜测语义。
int endToEndFlag(const ModelInfo& info) {
    std::smatch match;
    const auto direct = info.metadata.find("end2end");
    if (direct != info.metadata.end()) {
        if (!std::regex_match(direct->second, match, std::regex("\\s*(true|false|1|0)\\s*", std::regex::icase)))
            throw std::invalid_argument("ONNX end2end 元数据必须为 true/false/1/0");
        const auto value = match[1].str();
        return value == "1" || value[0] == 't' || value[0] == 'T' ? 1 : 0;
    }
    const auto args = info.metadata.find("args");
    if (args != info.metadata.end() && std::regex_search(args->second, match,
        std::regex("['\"]?end2end['\"]?\\s*:\\s*(true|false|1|0)\\b", std::regex::icase))) {
        const auto value = match[1].str();
        return value == "1" || value[0] == 't' || value[0] == 'T' ? 1 : 0;
    }
    return info.endToEnd ? 1 : -1;
}

/// 将灰度/BGR/BGRA 图等比例缩放并以 114 填边；记录逆变换参数，拒绝非 8 位输入。
Transform letterbox(const cv::Mat& source, cv::Size target) {
    if (source.empty() || source.depth() != CV_8U) throw std::invalid_argument("输入图片必须是非空 8 位图像");
    cv::Mat bgr;
    if (source.channels() == 1) cv::cvtColor(source, bgr, cv::COLOR_GRAY2BGR);
    else if (source.channels() == 4) cv::cvtColor(source, bgr, cv::COLOR_BGRA2BGR);
    else if (source.channels() == 3) bgr = source;
    else throw std::invalid_argument("输入图片必须为灰度、BGR 或 BGRA");
    Transform transform;
    transform.gain = std::min(target.width / static_cast<float>(source.cols), target.height / static_cast<float>(source.rows));
    transform.width = static_cast<int>(std::round(source.cols * transform.gain));
    transform.height = static_cast<int>(std::round(source.rows * transform.gain));
    if (transform.width < 1 || transform.height < 1) throw std::invalid_argument("图像宽高比过大，缩放后有效区域为空");
    transform.left = static_cast<int>(std::round((target.width - transform.width) / 2.0 - 0.1));
    transform.top = static_cast<int>(std::round((target.height - transform.height) / 2.0 - 0.1));
    cv::Mat resized;
    cv::resize(bgr, resized, {transform.width, transform.height}, 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, transform.image, transform.top, target.height - transform.height - transform.top,
        transform.left, target.width - transform.width - transform.left, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return transform;
}

/// 生成覆盖整个轴向的切片起点；最后一片贴边，避免遗漏右侧和底部。
std::vector<int> tileStarts(int length, int size, int step) {
    std::vector<int> result{0};
    if (length <= size) return result;
    for (int64_t position = step; position < static_cast<int64_t>(length) - size; position += step)
        result.push_back(static_cast<int>(position));
    if (result.back() != length - size) result.push_back(length - size);
    return result;
}

/// 计算两个浮点检测框的 IoU，零面积框返回零，用于类别内抑制。
float iou(const cv::Rect2f& first, const cv::Rect2f& second) {
    float intersection = (first & second).area();
    float sum = first.area() + second.area() - intersection;
    return sum > 0 ? intersection / sum : 0;
}

/// 用系数和原型重建 logits，去除 letterbox 后放大到原图，返回框内概率阈值掩码以降低常驻内存。
cv::Mat reconstructMask(const float* prototype, int channels, int height, int width,
    const std::vector<float>& coefficients, cv::Size original, cv::Rect box,
    double threshold) {
    cv::Mat coefficient(1, channels, CV_32F, const_cast<float*>(coefficients.data()));
    cv::Mat basis(channels, height * width, CV_32F, const_cast<float*>(prototype));
    cv::Mat flattened;
    cv::gemm(coefficient, basis, 1, cv::Mat(), 0, flattened);
    cv::Mat logits = flattened.reshape(1, height);
    // 与官方 process_mask_native/scale_masks 一致：在原型分辨率重新计算有效尺寸。
    // 直接缩小输入层已经取整的 padding，在奇数边长图像上会产生一列或一行的掩码偏移。
    const double gain = std::min(width / static_cast<double>(original.width), height / static_cast<double>(original.height));
    const int contentWidth = std::max(1, static_cast<int>(std::nearbyint(original.width * gain)));
    const int contentHeight = std::max(1, static_cast<int>(std::nearbyint(original.height * gain)));
    const int left = static_cast<int>(std::nearbyint((width - contentWidth) / 2.0 - 0.1));
    const int top = static_cast<int>(std::nearbyint((height - contentHeight) / 2.0 - 0.1));
    cv::Rect content = cv::Rect(left, top, contentWidth, contentHeight) & cv::Rect(0, 0, width, height);
    if (content.empty()) throw std::runtime_error("分割原型的有效区域为空");
    cv::Mat resized;
    cv::resize(logits(content), resized, original, 0, 0, cv::INTER_LINEAR);
    box &= cv::Rect(0, 0, original.width, original.height);
    if (box.empty()) return {};
    cv::Mat result = cv::Mat::zeros(box.size(), CV_8UC1);
    if (threshold >= 1) return result;
    const double logit = threshold <= 0 ? -std::numeric_limits<double>::infinity() : std::log(threshold / (1 - threshold));
    cv::compare(resized(box), logit, result, cv::CMP_GT);
    return result;
}
}

/// 创建版本与任务固定的实例，不在构造阶段访问 CUDA。
YoloRunner::YoloRunner(int version, bool segmentation) : version_(version), segmentation_(segmentation) {}
/// 通过独占智能指针释放 engine 及其资源，不影响其他模型。
YoloRunner::~YoloRunner() = default;

/// 串行构建 ONNX engine 并保存错误详情；引擎构建不替换当前已初始化的模型。
int YoloRunner::convert(const std::string& source, const std::string& destination, bool fp16) {
    std::lock_guard<std::mutex> guard(mutex_);
    try { buildEngine(source, destination, fp16); error_.clear(); return 0; }
    catch (const std::invalid_argument& error) { error_ = error.what(); return -2; }
    catch (const std::exception& error) { error_ = error.what(); return -1; }
}

/// 加载 engine 并验证批次、任务与类别；失败后本实例保持未初始化，错误可通过 error 读取。
int YoloRunner::init(const std::string& path, int classCount, int batchSize) {
    std::lock_guard<std::mutex> guard(mutex_);
    results_.clear(); engine_.reset(); input_.clear(); classes_ = 0;
    try {
        engine_ = std::make_unique<Engine>();
        engine_->load(path);
        inspect();
        const auto& info = engine_->info();
        if ((classCount > 0 && classCount != classes_) || (batchSize > 0 && batchSize != info.batch))
            throw std::invalid_argument("指定类别数或批次与 engine 不一致；传入 0 可自动读取");
        input_.resize(static_cast<size_t>(info.batch) * info.channels * info.height * info.width);
        error_.clear(); return 0;
    } catch (const std::invalid_argument& error) { error_ = error.what(); engine_.reset(); return -2; }
    catch (const std::exception& error) { error_ = error.what(); engine_.reset(); return -1; }
}

/// 分析预测及原型张量，按 YOLO 版本区分 objectness、通道优先及最终输出布局。
void YoloRunner::inspect() {
    const auto& info = engine_->info();
    if (info.channels != 3) throw std::invalid_argument("YOLO 输入必须为三通道 NCHW 图像");
    prediction_ = prototype_ = -1; masks_ = protoHeight_ = protoWidth_ = 0;
    for (size_t index = 0; index < info.outputs.size(); ++index) {
        const auto& shape = info.outputs[index].shape;
        if (shape.size() == 3) {
            if (prediction_ >= 0) throw std::invalid_argument("YOLO engine 包含多个预测张量，当前接口无法确定输出语义");
            prediction_ = static_cast<int>(index);
        } else if (shape.size() == 4) {
            if (prototype_ >= 0) throw std::invalid_argument("YOLO engine 包含多个分割原型张量");
            prototype_ = static_cast<int>(index);
        } else throw std::invalid_argument("YOLO 输出必须为预测张量 [B,N,C]/[B,C,N] 及可选分割原型 [B,M,H,W]");
    }
    if (prediction_ < 0 || (prototype_ >= 0) != segmentation_)
        throw std::invalid_argument("engine 检测/分割任务与所选接口不一致");
    if (segmentation_) {
        const auto& shape = info.outputs[prototype_].shape;
        if (shape[0] != info.batch) throw std::invalid_argument("分割原型批次与输入不一致");
        masks_ = shape[1]; protoHeight_ = shape[2]; protoWidth_ = shape[3];
    }
    const auto& shape = info.outputs[prediction_].shape;
    if (shape[0] != info.batch) throw std::invalid_argument("预测输出批次与输入不一致");
    // 两类别 raw 输出的通道数也等于 6+nm；必须优先使用分支元数据，不能仅凭 max_det 猜测。
    const int endFlag = endToEndFlag(info);
    const bool finalShape = shape[2] == 6 + masks_;
    if (endFlag == 1 && (version_ != 26 || !finalShape))
        throw std::invalid_argument("end2end 元数据与所选 YOLO 接口或最终输出形状不一致");
    endToEnd_ = version_ == 26 && finalShape && endFlag != 0;
    if (endToEnd_ && endFlag < 0 && info.classes == 2)
        throw std::invalid_argument("两类别 row-major 输出存在 raw/end2end 歧义，请在 ONNX 中保留明确的 end2end 元数据");
    // 缺少类别元数据时沿用各版本的标准导出布局；小图像的候选数可能小于类别通道数，不能以长短轴判断。
    transposed_ = !endToEnd_ && version_ != 5;
    if (!endToEnd_ && info.classes > 0) {
        const int expected = (version_ == 5 ? 5 : 4) + info.classes + masks_;
        if (shape[1] == expected && shape[2] == expected) transposed_ = version_ != 5;
        else if (shape[2] == expected) transposed_ = false;
        else if (shape[1] == expected) transposed_ = true;
        else throw std::invalid_argument("ONNX 类别元数据与预测输出通道数不一致");
    }
    channels_ = shape[transposed_ ? 1 : 2]; anchors_ = shape[transposed_ ? 2 : 1];
    classes_ = endToEnd_ ? info.classes : channels_ - (version_ == 5 ? 5 : 4) - masks_;
    if (classes_ < 1) throw std::invalid_argument("无法从输出获取类别数；YOLO26 end2end 模型必须保留 ONNX names/nc 元数据");
    if (info.classes > 0 && classes_ != info.classes) throw std::invalid_argument("ONNX 类别元数据与 engine 不一致");
}

/// 参数通过完整校验后才更新，非法输入保持此前配置并返回 -2。
int YoloRunner::setPara(int size, int overlapSize, double threshold) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (size <= 0 || overlapSize < 0 || overlapSize >= size || !std::isfinite(threshold) || threshold < 0 || threshold > 1) {
        error_ = "切片尺寸必须大于零、重叠须小于尺寸、阈值须在 [0,1]"; return -2;
    }
    cutSize_ = size; overlap_ = overlapSize; threshold_ = threshold; error_.clear(); return 0;
}

/// 仅清理本实例资源，允许重复释放及之后重新初始化。
int YoloRunner::dispose() {
    std::lock_guard<std::mutex> guard(mutex_);
    engine_.reset(); results_.clear(); input_.clear(); classes_ = 0; error_.clear(); return 0;
}

/// 按当前 engine 批次组织整图或切片推理，所有结果和掩码均映射到整张输入图像。
int YoloRunner::run(cv::Mat& image, int& len, bool tiles) {
    std::lock_guard<std::mutex> guard(mutex_);
    len = 0; results_.clear();
    if (!engine_) { error_ = "模型尚未初始化"; return -3; }
    try {
        if (image.empty() || image.depth() != CV_8U || (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
            throw std::invalid_argument("输入必须是非空 8 位灰度/BGR/BGRA 图片");
        if (!tiles) infer({image}, {cv::Point()}, image.size());
        else {
            std::vector<cv::Mat> batch;
            std::vector<cv::Point> offsets;
            const auto startsX = tileStarts(image.cols, cutSize_, cutSize_ - overlap_);
            const auto startsY = tileStarts(image.rows, cutSize_, cutSize_ - overlap_);
            for (int y : startsY) for (int x : startsX) {
                batch.push_back(image(cv::Rect(x, y, std::min(cutSize_, image.cols - x), std::min(cutSize_, image.rows - y))));
                offsets.emplace_back(x, y);
                if (batch.size() == static_cast<size_t>(engine_->info().batch)) {
                    infer(batch, offsets, image.size()); batch.clear(); offsets.clear();
                }
            }
            if (!batch.empty()) infer(batch, offsets, image.size());
        }
        if (results_.size() > static_cast<size_t>(std::numeric_limits<int>::max())) throw std::runtime_error("目标数量超过接口可表示范围");
        len = static_cast<int>(results_.size()); error_.clear(); return 0;
    } catch (const std::invalid_argument& error) { error_ = error.what(); results_.clear(); return -2; }
    catch (const std::exception& error) { error_ = error.what(); results_.clear(); return -1; }
}

/// 预处理有效图片、执行固定批次推理并进行同类 NMS；end2end 输出直接保留模型选出的实例。
void YoloRunner::infer(const std::vector<cv::Mat>& images, const std::vector<cv::Point>& offsets, cv::Size canvas) {
    const auto& info = engine_->info();
    const cv::Size target(info.width, info.height);
    const size_t planeSize = static_cast<size_t>(info.height) * info.width;
    std::fill(input_.begin(), input_.end(), 0.0f);
    std::vector<Transform> transforms;
    transforms.reserve(images.size());
    for (size_t batch = 0; batch < images.size(); ++batch) {
        transforms.push_back(letterbox(images[batch], target));
        cv::Mat rgb, normalized;
        cv::cvtColor(transforms.back().image, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
        std::vector<cv::Mat> planes;
        for (size_t channel = 0; channel < 3; ++channel)
            planes.emplace_back(info.height, info.width, CV_32F, input_.data() + (batch * 3 + channel) * planeSize);
        cv::split(normalized, planes);
    }
    auto tensors = engine_->infer(input_);
    const auto& prediction = tensors.at(prediction_).values;
    const std::vector<float>* prototype = segmentation_ ? &tensors.at(prototype_).values : nullptr;
    const size_t predictionStride = static_cast<size_t>(anchors_) * channels_;
    const size_t prototypeStride = static_cast<size_t>(masks_) * protoHeight_ * protoWidth_;
    if (prediction.size() != static_cast<size_t>(info.batch) * predictionStride ||
        (prototype && prototype->size() != static_cast<size_t>(info.batch) * prototypeStride))
        throw std::runtime_error("engine 返回的输出数据数量与初始化尺寸不一致");
    for (size_t batch = 0; batch < images.size(); ++batch) {
        const float* data = prediction.data() + batch * predictionStride;
        std::vector<Candidate> candidates;
        const float confidence = segmentation_ ? 0.5f : static_cast<float>(threshold_);
        for (int anchor = 0; anchor < anchors_; ++anchor) {
            // 通道优先输出无需额外转置和整块复制，直接按 stride 读取各分量。
            const auto value = [&](int channel) -> float { return data[transposed_ ? static_cast<size_t>(channel) * anchors_ + anchor : static_cast<size_t>(anchor) * channels_ + channel]; };
            float x = value(0), y = value(1), width = value(2), height = value(3);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(width) || !std::isfinite(height)) continue;
            int classId = -1;
            float score = -1;
            int coefficientStart;
            if (endToEnd_) {
                score = value(4);
                const float classValue = value(5);
                if (!std::isfinite(classValue) || classValue < 0 || classValue >= classes_ || std::floor(classValue) != classValue) continue;
                classId = static_cast<int>(classValue);
                width -= x; height -= y; coefficientStart = 6;
            } else {
                const int start = version_ == 5 ? 5 : 4;
                for (int current = 0; current < classes_; ++current) {
                    const float probability = value(start + current);
                    if (std::isfinite(probability) && probability >= 0 && probability <= 1.0001f && probability > score) {
                        score = probability; classId = current;
                    }
                }
                if (version_ == 5) {
                    const float objectness = value(4);
                    if (!std::isfinite(objectness) || objectness < 0 || objectness > 1.0001f) continue;
                    score *= objectness;
                }
                x -= width * 0.5f; y -= height * 0.5f; coefficientStart = start + classes_;
            }
            // 延续旧接口及官方 NMS 的严格大于阈值语义；阈值为零时也不能把零分背景当成目标。
            if (classId < 0 || !std::isfinite(score) || score <= confidence || score > 1.0001f || width <= 0 || height <= 0) continue;
            Candidate candidate{classId, score, {x, y, width, height}, {}};
            bool valid = true;
            for (int mask = 0; mask < masks_; ++mask) {
                float coefficient = value(coefficientStart + mask);
                if (!std::isfinite(coefficient)) { valid = false; break; }
                candidate.coefficients.push_back(coefficient);
            }
            if (valid) candidates.push_back(std::move(candidate));
        }
        std::stable_sort(candidates.begin(), candidates.end(), higherScore);
        std::vector<size_t> kept;
        for (size_t index = 0; index < candidates.size() && kept.size() < 300; ++index) {
            bool suppressed = false;
            if (!endToEnd_) for (size_t previous : kept)
                if (candidates[index].classId == candidates[previous].classId && iou(candidates[index].box, candidates[previous].box) > 0.45f) {
                    suppressed = true; break;
                }
            if (!suppressed) kept.push_back(index);
        }
        const auto& transform = transforms[batch];
        for (size_t index : kept) {
            const auto& candidate = candidates[index];
            const auto& box = candidate.box;
            const float left = std::clamp((box.x - transform.left) / transform.gain, 0.0f, static_cast<float>(images[batch].cols));
            const float top = std::clamp((box.y - transform.top) / transform.gain, 0.0f, static_cast<float>(images[batch].rows));
            const float right = std::clamp((box.x + box.width - transform.left) / transform.gain, 0.0f, static_cast<float>(images[batch].cols));
            const float bottom = std::clamp((box.y + box.height - transform.top) / transform.gain, 0.0f, static_cast<float>(images[batch].rows));
            cv::Rect local(static_cast<int>(std::floor(left)), static_cast<int>(std::floor(top)),
                static_cast<int>(std::ceil(right)) - static_cast<int>(std::floor(left)),
                static_cast<int>(std::ceil(bottom)) - static_cast<int>(std::floor(top)));
            if (local.empty()) continue;
            YoloResult result{candidate.classId, candidate.score, local + offsets[batch], {}, canvas};
            if (prototype) {
                result.mask = reconstructMask(prototype->data() + batch * prototypeStride, masks_, protoHeight_, protoWidth_,
                    candidate.coefficients, images[batch].size(), local, threshold_);
            }
            results_.push_back(std::move(result));
        }
    }
}

/// 拷贝原图目标结果；先校验全部输出指针，分割掩码返回独立副本以隔离后续推理。
int YoloRunner::result(int index, int* classId, float* score, float* x, float* y, float* w, float* h, cv::Mat* mask) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!engine_) { error_ = "模型尚未初始化"; return -3; }
    if (index < 0 || static_cast<size_t>(index) >= results_.size() || !classId || !score || !x || !y || !w || !h || (segmentation_ && !mask)) {
        error_ = "结果下标越界或输出指针为空"; return -2;
    }
    try {
        const auto& result = results_[index];
        if (mask) {
            // 内部保留框内数据即可；公开接口仍返回完整原图画布的独立掩码。
            cv::Mat full = cv::Mat::zeros(result.canvas, CV_8UC1);
            result.mask.copyTo(full(result.box));
            *mask = std::move(full);
        }
        *classId = result.classId; *score = result.score; *x = static_cast<float>(result.box.x); *y = static_cast<float>(result.box.y);
        *w = static_cast<float>(result.box.width); *h = static_cast<float>(result.box.height);
        error_.clear(); return 0;
    } catch (const std::exception& error) { error_ = error.what(); return -1; }
}

/// 返回错误字符串副本，锁内读取避免与推理线程并发访问字符串。
std::string YoloRunner::error() const { std::lock_guard<std::mutex> guard(mutex_); return error_; }

/// 获取 engine 自动识别的批次、输入尺寸与类别数，未初始化时不写出参数。
int YoloRunner::modelInfo(int& batch, int& width, int& height, int& classes) const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!engine_) return -3;
    const auto& info = engine_->info(); batch = info.batch; width = info.width; height = info.height; classes = classes_; return 0;
}

/// 明确拒绝已弃用的 WTS/架构配置入口，将迁移方式保存到本实例错误信息。
int YoloRunner::retired(const std::string& message) {
    std::lock_guard<std::mutex> guard(mutex_); error_ = message; return -2;
}
}
