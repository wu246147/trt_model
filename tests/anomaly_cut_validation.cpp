#include "anomaly_trt.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
/// 失败时抛出便于定位的测试错误，避免静默通过。
void check(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }
/// 在模型操作完成后取得错误详情，避免实参求值顺序提前读取空错误。
void status(AnomalyTRT& model, int code) { if (code != 0) throw std::runtime_error(model.lastError()); }
/// 根据测试参数选择模型类型。
AnomalyKind kind(const std::string& name) {
    if (name == "efficientad") return AnomalyKind::EfficientAD;
    if (name == "anomalydino") return AnomalyKind::AnomalyDINO;
    if (name == "dinomaly") return AnomalyKind::Dinomaly;
    throw std::runtime_error("异常模型参数无效");
}
/// 获取模型的独立结果副本，便于在下一次推理后继续比较。
AnomalyResult result(AnomalyTRT& model) { AnomalyResult value; status(model, model.getResult(value)); return value; }
}

/// 验证真实引擎切片的单片等价、重复图一致、重叠max/OR融合、边缘覆盖和错误恢复。
int main(int argc, char** argv) {
    try {
        check(argc == 4, "用法: anomaly_cut_validation 模型名 engine 样本图片");
        AnomalyTRT model(kind(argv[1])); status(model, model.initModel(argv[2]));
        auto image = cv::imread(argv[3]); check(!image.empty(), "无法读取切图验证样本");
        const auto thresholds = model.modelInfo(); int len = 0;
        status(model, model.runModel(image, len)); const auto whole = result(model);
        status(model, model.setCutPara(std::max(image.rows, image.cols) + 1, 0));
        status(model, model.runCutModel(image, len)); const auto oneTile = result(model);
        check(len == 1 && oneTile.mask.size() == image.size() && oneTile.overlay.size() == image.size(), "单片输出尺寸或len错误");
        check(std::abs(oneTile.score - whole.score) < 1e-6 && cv::norm(oneTile.anomalyMap, whole.anomalyMap, cv::NORM_INF) < 1e-6 && cv::countNonZero(oneTile.mask != whole.mask) == 0, "整图与单切片推理不等价");
        const int side = std::min({97, image.cols, image.rows}); auto square = image(cv::Rect(0, 0, side, side));
        status(model, model.runModel(square, len)); const auto base = result(model);
        cv::Mat repeated; cv::repeat(square, 2, 3, repeated);
        status(model, model.setPara(side, 0, -1)); status(model, model.runModelCutImage(repeated, len)); const auto repeatedResult = result(model);
        cv::Mat repeatedMap, repeatedMask; cv::repeat(base.anomalyMap, 2, 3, repeatedMap); cv::repeat(base.mask, 2, 3, repeatedMask);
        check(std::abs(base.score - repeatedResult.score) < 1e-6 && cv::norm(repeatedMap, repeatedResult.anomalyMap, cv::NORM_INF) < 1e-6 && cv::countNonZero(repeatedMask != repeatedResult.mask) == 0, "重复切片图的得分、坐标或掩码不一致");
        cv::Mat irregular; cv::resize(image, irregular, cv::Size(257, 193));
        status(model, model.setCutPara(128, 33)); status(model, model.runCutModel(irregular, len)); const auto merged = result(model);
        cv::Mat expectedMap = cv::Mat::zeros(irregular.size(), CV_32FC1), expectedMask = cv::Mat::zeros(irregular.size(), CV_8UC1), coverage = cv::Mat::zeros(irregular.size(), CV_8UC1);
        float expectedScore = 0;
        // 用独立整图接口逐块推理验证融合；固定非整倍尺寸同时覆盖重叠区和右下不足整片的边界。
        for (int y : {0, 95}) for (int x : {0, 95, 190}) {
            const cv::Rect region(x, y, std::min(128, irregular.cols - x), std::min(128, irregular.rows - y));
            auto tile = irregular(region); status(model, model.runModel(tile, len)); const auto tileResult = result(model);
            auto mapRegion = expectedMap(region); auto maskRegion = expectedMask(region);
            cv::max(mapRegion, tileResult.anomalyMap, mapRegion); cv::bitwise_or(maskRegion, tileResult.mask, maskRegion);
            coverage(region).setTo(255); expectedScore = std::max(expectedScore, tileResult.score);
        }
        const double mapError = cv::norm(merged.anomalyMap, expectedMap, cv::NORM_INF);
        check(cv::countNonZero(coverage) == irregular.rows * irregular.cols, "独立切片验证存在未覆盖像素");
        check(mapError < 1e-6 && cv::countNonZero(merged.mask != expectedMask) == 0 && std::abs(merged.score - expectedScore) < 1e-6, "重叠异常图max、mask OR或score max融合错误");
        check(model.setCutPara(0, 0) == -2 && model.setCutPara(128, 128) == -2 && model.setCutPara(128, -1) == -2, "无效切片参数没有被拒绝");
        const auto current = model.modelInfo();
        check(current.imageThreshold == thresholds.imageThreshold && current.pixelThreshold == thresholds.pixelThreshold, "切图操作意外修改阈值");
        cv::Mat empty; check(model.runCutModel(empty, len) != 0 && len == 0, "空图切片没有返回失败");
        AnomalyResult stale; check(model.getResult(stale) == -3, "失败后仍保留过期切片结果");
        status(model, model.runCutModel(irregular, len)); check(len == 1, "切片错误后无法恢复");
        check(RaivasAnomalyTRT_Create(-1) == nullptr && RaivasAnomalyTRT_DetectCut(nullptr, &image, &stale) == -2, "C接口未拒绝无效参数");
        std::cout << "{\"model\":\"" << argv[1] << "\",\"passed\":true,\"manual_fusion_max_error\":" << mapError << ",\"edge_coverage\":1,\"repeat_score_error\":" << std::abs(base.score - repeatedResult.score) << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
