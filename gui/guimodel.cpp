#include "guimodel.h"
#include "anomaly_trt.h"
#include "yolov5_det.h"
#include "yolov5_seg.h"
#include "yolo11_det.h"
#include "yolo11_seg.h"
#include "yolo26_det.h"
#include "yolo26_seg.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

/// 将调用失败及模型保存的原始错误合并为可显示的异常。
void checkResult(int code, const std::string& error, const char* operation) {
    if (code != 0)
        throw std::runtime_error(std::string(operation) + "：" + error + "（返回码 " + std::to_string(code) + "）");
}

/// 统一为 BGR 图像副本，绘制时不修改用户加载的原图。
cv::Mat bgrImage(const cv::Mat& image) {
    cv::Mat result;
    if (image.channels() == 1) cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
    else if (image.channels() == 4) cv::cvtColor(image, result, cv::COLOR_BGRA2BGR);
    else result = image.clone();
    return result;
}

/// 为不同类别生成确定颜色，保证同一类别在多张图像上保持一致。
cv::Scalar classColor(int classId) {
    static const cv::Scalar palette[] = {
        {80, 220, 80}, {40, 160, 255}, {255, 150, 70},
        {220, 100, 220}, {60, 230, 240}, {240, 210, 60}
    };
    return palette[static_cast<unsigned int>(classId) % 6];
}

/// 用模板复用六种 YOLO 的相同接口，Seg 在编译期选择是否读取掩码。
template<class Model, bool Seg>
class YoloGuiModel final : public GuiModel {
public:
    /// 转换 ONNX 并在失败时提供底层 TensorRT 错误信息。
    void build(const QString& onnx, const QString& engine, bool fp16) override {
        const int code = model_.onnxToEngine(onnx.toStdString(), engine.toStdString(), fp16);
        checkResult(code, model_.getLastError(), "ONNX 转换失败");
    }

    /// 从引擎自动读取输入、批次和类别信息，不使用界面猜测的参数。
    GuiModelInfo load(const QString& path) override {
        const int code = model_.initModel(path.toStdString());
        checkResult(code, model_.getLastError(), "模型初始化失败");
        GuiModelInfo info;
        const int infoCode = model_.getModelInfo(info.batch, info.width, info.height, info.classes);
        checkResult(infoCode, model_.getLastError(), "读取模型信息失败");
        info.details = QString("批次 %1；输入 %2 × %3；类别 %4")
            .arg(info.batch).arg(info.width).arg(info.height).arg(info.classes);
        return info;
    }

    /// 推理整图或切片，将原图坐标的框与实例掩码绘制到独立结果快照中。
    GuiPrediction run(const cv::Mat& image, double threshold, bool tiles, int tileSize, int overlap) override {
        const int configCode = model_.setPara(tiles ? tileSize : 640, tiles ? overlap : 64, threshold);
        checkResult(configCode, model_.getLastError(), "设置推理参数失败");
        cv::Mat input = image;
        int count = 0;
        const int code = tiles ? model_.runModelCutImage(input, count) : model_.runModel(input, count);
        checkResult(code, model_.getLastError(), "模型推理失败");
        GuiPrediction prediction;
        prediction.overlay = bgrImage(image);
        prediction.count = count;
        if constexpr (Seg) prediction.mask = cv::Mat::zeros(image.size(), CV_8UC1);
        for (int i = 0; i < count; ++i) {
            int classId = 0;
            float score = 0, x = 0, y = 0, width = 0, height = 0;
            cv::Mat mask;
            int readCode = 0;
            if constexpr (Seg)
                readCode = model_.getDefectDetResult(i, &classId, &score, &x, &y, &width, &height, mask);
            else
                readCode = model_.getDefectDetResult(i, &classId, &score, &x, &y, &width, &height);
            checkResult(readCode, model_.getLastError(), "读取模型结果失败");
            const cv::Scalar color = classColor(classId);
            if constexpr (Seg) {
                if (!mask.empty()) {
                    cv::Mat colored(image.size(), CV_8UC3, color), blended;
                    cv::addWeighted(prediction.overlay, 0.6, colored, 0.4, 0, blended);
                    blended.copyTo(prediction.overlay, mask);
                    cv::bitwise_or(prediction.mask, mask, prediction.mask);
                }
            }
            const cv::Rect box = cv::Rect(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)),
                std::max(0, static_cast<int>(std::round(width))), std::max(0, static_cast<int>(std::round(height))))
                & cv::Rect(0, 0, image.cols, image.rows);
            if (box.empty()) continue;
            cv::rectangle(prediction.overlay, box, color, 2);
            const std::string label = "ID " + std::to_string(classId) + " " + cv::format("%.3f", score);
            int baseline = 0;
            const cv::Size text = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
            const int labelWidth = std::min(image.cols, text.width + 8);
            const int labelHeight = std::min(image.rows, text.height + baseline + 6);
            const int labelX = std::min(box.x, image.cols - labelWidth);
            const int labelY = box.y >= labelHeight ? box.y - labelHeight : std::min(box.y, image.rows - labelHeight);
            // 高对比度底色让类别和得分在复杂背景与缩小显示时仍可辨认。
            cv::rectangle(prediction.overlay, cv::Rect(labelX, labelY, labelWidth, labelHeight), color, cv::FILLED);
            cv::putText(prediction.overlay, label, cv::Point(labelX + 4, labelY + text.height + 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
        }
        prediction.details = QString("%1 个实例").arg(count);
        return prediction;
    }
private:
    Model model_;
};

/// 将三个异常模型适配为同一界面流程，保持模型各自的校准判定阈值。
class AnomalyGuiModel final : public GuiModel {
public:
    /// 根据模型类型创建独立异常检测实例。
    explicit AnomalyGuiModel(AnomalyKind kind) : model_(kind) {}

    /// 使用共享 TensorRT 转换流程生成引擎并保留原始失败原因。
    void build(const QString& onnx, const QString& engine, bool fp16) override {
        const int code = model_.onnxToEngine(onnx.toStdString(), engine.toStdString(), fp16);
        checkResult(code, model_.lastError(), "异常模型 ONNX 转换失败");
    }

    /// 读取模型的实际网络尺寸、批次、类别名称和校准阈值。
    GuiModelInfo load(const QString& path) override {
        const int code = model_.initModel(path.toStdString());
        checkResult(code, model_.lastError(), "异常模型初始化失败");
        const auto value = model_.modelInfo();
        GuiModelInfo info;
        info.batch = value.batchSize;
        info.width = value.networkSize.width;
        info.height = value.networkSize.height;
        info.classes = 2;
        info.details = QString("批次 %1；输入 %2 × %3\n类别：%4 / %5\n模型图像阈值 %6；像素阈值 %7")
            .arg(info.batch).arg(info.width).arg(info.height)
            .arg(QString::fromStdString(value.normalClass), QString::fromStdString(value.anomalyClass))
            .arg(value.imageThreshold, 0, 'g', 5).arg(value.defaultPixelThreshold, 0, 'g', 5);
        return info;
    }

    /// 按整图或切片执行异常检测；像素阈值可覆盖，整图判定保留模型校准阈值，返回原图大小的融合结果。
    GuiPrediction run(const cv::Mat& image, double threshold, bool tiles, int tileSize, int overlap) override {
        const int parameterCode = model_.setPara(-1, threshold);
        checkResult(parameterCode, model_.lastError(), "设置异常模型阈值失败");
        if (tiles) {
            const int cutCode = model_.setCutPara(tileSize, overlap);
            checkResult(cutCode, model_.lastError(), "设置异常模型切片参数失败");
        }
        cv::Mat input = image;
        int count = 0;
        const int code = tiles ? model_.runCutModel(input, count) : model_.runModel(input, count);
        checkResult(code, model_.lastError(), "异常模型推理失败");
        AnomalyResult result;
        const int resultCode = model_.getResult(result);
        checkResult(resultCode, model_.lastError(), "读取异常结果失败");
        GuiPrediction prediction;
        prediction.overlay = result.overlay;
        prediction.mask = result.mask;
        prediction.heatmap = result.heatmap;
        prediction.count = count;
        prediction.details = QString("%1；图像得分 %2；像素异常面积 %3")
            .arg(QString::fromStdString(result.className)).arg(result.score, 0, 'f', 5)
            .arg(result.mask.empty() ? 0 : cv::countNonZero(result.mask));
        return prediction;
    }
private:
    AnomalyTRT model_;
};
}

/// 返回测试项名称，越界枚举返回明确的未知任务标记。
QString guiTaskName(GuiTask task) {
    static const char* names[] = {"YOLOv5 检测", "YOLOv5 实例分割", "YOLO11 检测", "YOLO11 实例分割",
        "YOLO26 检测", "YOLO26 实例分割", "AnomalyDINO 异常检测", "Dinomaly 异常检测", "EfficientAD 异常检测"};
    const int index = static_cast<int>(task);
    return index >= 0 && index < 9 ? QString::fromUtf8(names[index]) : QString("未知任务");
}

/// 后三项为异常检测，各模型的判定和掩码阈值语义相同。
bool isAnomalyTask(GuiTask task) { return static_cast<int>(task) >= 6; }

/// 三个 YOLO 分割入口输出实例掩码。
bool isSegmentationTask(GuiTask task) {
    const int index = static_cast<int>(task);
    return index < 6 && index % 2 == 1;
}

/// 根据选中的任务构造独立适配器，不共享默认 C 导出实例。
std::shared_ptr<GuiModel> createGuiModel(GuiTask task) {
    switch (task) {
    case GuiTask::Yolo5Det: return std::make_shared<YoloGuiModel<RaivasDefectDet5::RaivasDefectDet5, false>>();
    case GuiTask::Yolo5Seg: return std::make_shared<YoloGuiModel<RaivasDefectSeg5::RaivasDefectSeg5, true>>();
    case GuiTask::Yolo11Det: return std::make_shared<YoloGuiModel<RaivasDefectDet11::RaivasDefectDet11, false>>();
    case GuiTask::Yolo11Seg: return std::make_shared<YoloGuiModel<RaivasDefectSeg11::RaivasDefectSeg11, true>>();
    case GuiTask::Yolo26Det: return std::make_shared<YoloGuiModel<RaivasDefectDet26::RaivasDefectDet26, false>>();
    case GuiTask::Yolo26Seg: return std::make_shared<YoloGuiModel<RaivasDefectSeg26::RaivasDefectSeg26, true>>();
    case GuiTask::AnomalyDino: return std::make_shared<AnomalyGuiModel>(AnomalyKind::AnomalyDINO);
    case GuiTask::Dinomaly: return std::make_shared<AnomalyGuiModel>(AnomalyKind::Dinomaly);
    case GuiTask::EfficientAd: return std::make_shared<AnomalyGuiModel>(AnomalyKind::EfficientAD);
    }
    throw std::invalid_argument("不支持的界面测试任务");
}
