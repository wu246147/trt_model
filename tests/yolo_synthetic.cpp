#include "yolov5_det.h"
#include "yolov5_seg.h"
#include "yolo11_det.h"
#include "yolo11_seg.h"
#include "yolo26_det.h"
#include "yolo26_seg.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
/// 校验条件并抛出明确错误；不使用 Release 构建会禁用的 assert。
void require(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }

/// 校验单个模型的自动参数、解码、NMS、批次填充、坐标、分割及错误恢复。
template<class Model, bool Segmentation>
void check(const std::filesystem::path& directory, int version) {
    const std::string stem = "yolo" + std::to_string(version) + (Segmentation ? "_seg" : "_det");
    const std::string onnx = (directory / (stem + ".onnx")).string();
    const std::string engine = (directory / (stem + ".engine")).string();
    Model model;
    int batch = 0, width = 0, height = 0, classes = 0, count = 99;
    cv::Mat source(32, 96, CV_8UC3, cv::Scalar(30, 80, 150));
    require(model.getModelInfo(batch, width, height, classes) == -3, stem + ": 未初始化信息接口");
    require(model.runModel(source, count) == -3 && count == 0, stem + ": 未初始化检测");
    require(model.onnxToEngine(onnx, engine, false) == 0, stem + ": 转换失败 " + model.getLastError());
    require(model.initModel(engine, 3, 0) == -2, stem + ": 类别不匹配未报错");
    require(model.initModel(engine, 0, 1) == -2, stem + ": 批次不匹配未报错");
    require(model.initModel(engine) == 0, stem + ": 初始化失败 " + model.getLastError());
    require(model.getModelInfo(batch, width, height, classes) == 0 && batch == 2 && width == 96 && height == 64 && classes == 2,
        stem + ": 自动参数不正确");
    require(model.setPara(48, 48, 0.5) == -2, stem + ": 非法重叠未报错");
    require(model.setPara(48, 0, 0.5) == 0, stem + ": 参数设置失败");
    const int expected = version == 26 ? 3 : 2;
    require(model.runModel(source, count) == 0 && count == expected,
        stem + ": 解码/NMS/固定批次填充错误，数量=" + std::to_string(count) + " " + model.getLastError());
    for (int index = 0; index < count; ++index) {
        int classId = -1;
        float score = 0, x = 0, y = 0, w = 0, h = 0;
        cv::Mat mask;
        int code;
        if constexpr (Segmentation) code = model.getDefectDetResult(index, &classId, &score, &x, &y, &w, &h, mask);
        else code = model.getDefectDetResult(index, &classId, &score, &x, &y, &w, &h);
        require(code == 0 && classId >= 0 && classId < 2, stem + ": 目标读取失败");
        require(std::abs(x - 32) <= 1 && std::abs(y - 4) <= 1 && std::abs(w - 32) <= 1 && std::abs(h - 24) <= 1,
            stem + ": 矩形输入 letterbox 逆变换错误");
        if (index == 0) require(std::abs(score - (version == 5 ? 0.72f : 0.9f)) < 1e-4, stem + ": objectness/置信度计算错误");
        if constexpr (Segmentation) {
            require(mask.type() == CV_8UC1 && mask.size() == source.size(), stem + ": 掩码类型或大小错误");
            require(cv::countNonZero(mask) == 32 * 24 && mask.at<unsigned char>(16, 48) == 255,
                stem + ": 原型重建或掩码裁剪错误");
            mask.setTo(0);
            require(model.getDefectDetResult(index, &classId, &score, &x, &y, &w, &h, mask) == 0 && cv::countNonZero(mask) == 32 * 24,
                stem + ": 掩码未返回独立副本");
        }
    }
    require(model.runModelCutImage(source, count) == 0 && count == expected * 2, stem + ": 固定 batch=2 切片推理错误");
    for (int channels : {1, 4}) {
        cv::Mat image(32, 100, CV_MAKETYPE(CV_8U, channels), cv::Scalar::all(90));
        cv::Mat roi = image(cv::Rect(2, 0, 96, 32));
        require(!roi.isContinuous(), stem + ": 测试 ROI 应当非连续");
        require(model.runModel(roi, count) == 0 && count == expected, stem + ": 灰度/BGRA 或非连续 ROI 支持失败");
    }
    cv::Mat empty;
    require(model.runModel(empty, count) == -2 && count == 0 && !model.getLastError().empty(), stem + ": 空图错误状态不正确");
    require(model.runModel(source, count) == 0 && count == expected && model.getLastError().empty(), stem + ": 错误后恢复失败");
    require(model.disposeModel() == 0 && model.disposeModel() == 0, stem + ": 重复释放失败");
    require(model.runModel(source, count) == -3 && count == 0, stem + ": 释放后状态错误");
    std::cout << stem << ": passed\n";
}
}

/// 执行六种合成 engine 的完整公开接口测试；参数是 make_yolo_synthetic.py 的输出目录。
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("用法：yolo_synthetic <合成 ONNX 目录>");
        std::filesystem::path directory(argv[1]);
        check<RaivasDefectDet5::RaivasDefectDet5, false>(directory, 5);
        check<RaivasDefectSeg5::RaivasDefectSeg5, true>(directory, 5);
        check<RaivasDefectDet11::RaivasDefectDet11, false>(directory, 11);
        check<RaivasDefectSeg11::RaivasDefectSeg11, true>(directory, 11);
        check<RaivasDefectDet26::RaivasDefectDet26, false>(directory, 26);
        check<RaivasDefectSeg26::RaivasDefectSeg26, true>(directory, 26);
        require(RaivasDet11_Init((directory / "yolo11_det.engine").string().c_str()) == 0, "C 导出初始化失败");
        int batch = 0, width = 0, height = 0, classes = 0;
        require(RaivasDet11_GetModelInfo(&batch, &width, &height, &classes) == 0 && batch == 2 && classes == 2, "C 导出模型信息错误");
        require(RaivasDet11_GetModelInfo(nullptr, &width, &height, &classes) == -2, "C 导出空输出指针未校验");
        RaivasDet11_Destroy();
        std::cout << "All YOLO synthetic tests passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
