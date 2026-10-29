#include "anomaly_trt.h"
#include "anomaly_onnx_prepare.h"
#include "anomaly_preprocess.h"
#include "trt_engine.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
/// 验证条件并给出具体失败原因，用于开发期数值对比。
void check(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }

/// 在操作已经执行后查询错误，避免C++实参求值顺序导致提前读取空错误。
void checkStatus(AnomalyTRT& model, int status) { if (status != 0) throw std::runtime_error(model.lastError()); }

/// 读取二进制文件，路径采用UTF-8编码。
std::vector<char> readFile(const std::string& path) {
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    check(bool(file), "无法读取文件: " + path); const auto length = file.tellg(); check(length > 0, "输入文件为空");
    std::vector<char> bytes(static_cast<size_t>(length)); file.seekg(0); check(bool(file.read(bytes.data(), bytes.size())), "文件读取不完整"); return bytes;
}

/// 写入开发验证文件，失败不返回成功。
void writeFile(const std::string& path, const char* bytes, size_t size) {
    std::ofstream file(std::filesystem::u8path(path), std::ios::binary); check(bool(file), "无法创建文件: " + path);
    file.write(bytes, static_cast<std::streamsize>(size)); check(bool(file), "文件写入失败");
}

/// 根据命令行模型名称创建对应枚举。
AnomalyKind kind(const std::string& name) {
    if (name == "anomalydino") return AnomalyKind::AnomalyDINO;
    if (name == "dinomaly") return AnomalyKind::Dinomaly;
    if (name == "efficientad") return AnomalyKind::EfficientAD;
    throw std::runtime_error("模型参数应为anomalydino/dinomaly/efficientad");
}
}

/// 开发验证入口：build构建引擎，prepare导出兼容图，preprocess导出缩放张量，infer逐项核对参考结果并立即刷新报告输出。
int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    try {
        check(argc >= 2, "用法: prepare 输入.onnx 输出.onnx 高 宽 | preprocess 图片 宽 高 bilinear/bicubic 输出.bin | infer 模型名 engine 图片 expected.yml");
        const std::string mode = argv[1];
        if (mode == "info") {
            check(argc == 3, "info需要engine路径"); trt_model::Engine engine; engine.load(argv[2]);
            const auto& info = engine.info(); std::cout << info.batch << ',' << info.channels << ',' << info.height << ',' << info.width << '\n';
            for (const auto& item : info.outputs) { std::cout << item.name << ':'; for (auto value : item.shape) std::cout << value << ','; std::cout << '\n'; } return 0;
        }
        if (mode == "build") {
            check(argc == 5 || argc == 6, "build需要模型名、ONNX路径、engine路径和可选fp16标记");
            AnomalyTRT model(kind(argv[2]));
            checkStatus(model, model.onnxToEngine(argv[3], argv[4], argc == 6 && std::string(argv[5]) == "fp16")); return 0;
        }
        if (mode == "prepare") {
            check(argc == 6, "prepare需要输入ONNX、输出ONNX、高和宽");
            auto bytes = readFile(argv[2]);
            std::map<std::string, std::string> metadata{{"anomalib.model_type", "validation"}, {"anomalib.contract_version", "1"}, {"anomalib.preprocessing", "embedded"}, {"anomalib.resize_height", argv[4]}, {"anomalib.resize_width", argv[5]}};
            check(trt_model::prepareAnomalyOnnx(bytes, metadata), "兼容转换没有执行"); writeFile(argv[3], bytes.data(), bytes.size()); return 0;
        }
        if (mode == "preprocess") {
            check(argc == 7, "preprocess需要图片、宽、高、插值方式和输出文件");
            const auto encoded = readFile(argv[2]); const auto image = cv::imdecode(encoded, cv::IMREAD_COLOR);
            check(!image.empty(), "无法解码图片"); const std::string interpolation = argv[5]; check(interpolation == "bilinear" || interpolation == "bicubic", "插值方式无效");
            const auto input = trt_model::anomalyInput(image, cv::Size(std::stoi(argv[3]), std::stoi(argv[4])), interpolation == "bicubic", true);
            writeFile(argv[6], reinterpret_cast<const char*>(input.data()), input.size() * sizeof(float)); return 0;
        }
        check(mode == "infer" && argc == 6, "infer需要模型名、engine、图片和expected.yml");
        AnomalyTRT model(kind(argv[2])); checkStatus(model, model.initModel(argv[3]));
        const auto encoded = readFile(argv[4]); auto image = cv::imdecode(encoded, cv::IMREAD_COLOR); check(!image.empty(), "无法解码图片");
        int len = 0; checkStatus(model, model.runModel(image, len)); check(len == 1, "异常检测len应为1");
        AnomalyResult result; checkStatus(model, model.getResult(result));
        cv::FileStorage expected(argv[5], cv::FileStorage::READ); check(expected.isOpened(), "无法读取golden结果");
        float score = 0; int label = -1; cv::Mat networkMap, networkMask, map, mask;
        expected["score"] >> score; expected["label"] >> label; expected["anomaly_map"] >> networkMap; expected["mask"] >> networkMask;
        cv::resize(networkMap, map, image.size(), 0, 0, cv::INTER_LINEAR); cv::resize(networkMask, mask, image.size(), 0, 0, cv::INTER_NEAREST);
        const double scoreError = std::abs(result.score - score), mapError = cv::norm(result.anomalyMap, map, cv::NORM_INF);
        const double mapMean = cv::norm(result.anomalyMap, map, cv::NORM_L1) / map.total();
        const double agreement = 1.0 - static_cast<double>(cv::countNonZero(result.mask != mask)) / mask.total();
        std::cout << "{\"score\":" << result.score << ",\"label\":" << result.classId << ",\"score_abs_error\":" << scoreError << ",\"map_max_abs_error\":" << mapError << ",\"map_mean_abs_error\":" << mapMean << ",\"mask_agreement\":" << agreement << ",\"elapsed_ms\":" << result.elapsedMs << "}\n";
        // FP32允许GPU归约的微小差异，同时要求图片判定一致且掩码近乎一致。
        check(result.classId == label && scoreError < 0.005 && mapError < 0.02 && agreement > 0.995, "TensorRT与训练模型的数值或判定偏差超过容限");
        const auto info = model.modelInfo(); check(info.batchSize == 1, "自动读取batch失败");
        check(model.setPara(-2, 0.5) == -2, "未拒绝无效阈值");
        const auto savedMask = result.mask.clone(); result.mask.setTo(0); AnomalyResult copy;
        check(model.getResult(copy) == 0 && cv::norm(copy.mask, savedMask, cv::NORM_INF) == 0, "结果未执行独立深拷贝");
        model.disposeModel(); check(model.runModel(image, len) == -3 && len == 0, "未初始化状态处理错误");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
