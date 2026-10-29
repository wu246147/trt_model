#include "yolov5_det.h"
#include "yolo_runner.h"
namespace {
/// 每个 DLL 的 C 导出独占一个默认执行器，与 C++ 模型实例隔离。
trt_model::YoloRunner& apiModel() {
    static trt_model::YoloRunner model(5, false);
    return model;
}
}
namespace RaivasDefectDet5 {
struct RaivasDefectDet5::Impl { trt_model::YoloRunner runner{5, false}; };
/// 分配仅属于本对象的内部执行器，不访问 GPU。
RaivasDefectDet5::RaivasDefectDet5() : impl_(new Impl) {}
/// 销毁本对象的执行器及其模型资源。
RaivasDefectDet5::~RaivasDefectDet5() { delete impl_; }
/// 校验并保存切片大小、重叠和任务阈值。
int RaivasDefectDet5::setPara(int size, int overlap, double threshold) { return impl_->runner.setPara(size, overlap, threshold); }
/// 加载 engine 并自动读取参数；显式给定参数时验证一致性。
int RaivasDefectDet5::initModel(std::string path, int classes, int batch) { return impl_->runner.init(path, classes, batch); }
/// 调用原生 ONNX 解析器构建 engine，错误保存至本实例。
int RaivasDefectDet5::onnxToEngine(std::string source, std::string destination, bool fp16) { return impl_->runner.convert(source, destination, fp16); }
/// 释放本实例模型和结果，允许重复释放。
int RaivasDefectDet5::disposeModel() { return impl_->runner.dispose(); }
/// 整图推理并返回目标个数；失败清空结果。
int RaivasDefectDet5::runModel(cv::Mat& image, int& len) { return impl_->runner.run(image, len); }
/// 按切片配置推理，输出坐标和掩码映射回整图。
int RaivasDefectDet5::runModelCutImage(cv::Mat& image, int& len) { return impl_->runner.run(image, len, true); }
/// 兼容原 YOLO5 的切片方法名，复用相同执行流程。
int RaivasDefectDet5::runCutModel(cv::Mat& image, int& len) { return impl_->runner.run(image, len, true); }
/// 读取已缓存目标；分割任务同时返回独立掩码副本。
int RaivasDefectDet5::getDefectDetResult(int index, int* classId, float* score, float* x, float* y, float* w, float* h) { return impl_->runner.result(index, classId, score, x, y, w, h); }
/// 返回本实例自动识别的模型参数。
int RaivasDefectDet5::getModelInfo(int& batch, int& width, int& height, int& classes) const { return impl_->runner.modelInfo(batch, width, height, classes); }
/// 在线程锁保护下复制最近的错误说明。
std::string RaivasDefectDet5::getLastError() const { return impl_->runner.error(); }
/// 拒绝旧 WTS 转换，提示使用不需要手写网络结构的 ONNX 入口。
int RaivasDefectDet5::wtsToEngine(std::string, std::string, std::string, int) { return impl_->runner.retired("WTS 转换已废弃，请使用 onnxToEngine(onnxPath, enginePath)"); }
/// 命名空间级转换入口使用默认实例记录构建错误。
int onnxToEngine(std::string source, std::string destination, bool fp16) { return apiModel().convert(source, destination, fp16); }
/// 旧命名空间 WTS 入口保留符号并明确报错。
int wtsToEngine(std::string, std::string, std::string, int) { return apiModel().retired("WTS 转换已废弃，请使用 onnxToEngine(onnxPath, enginePath)"); }
}
/// 释放默认实例持有的模型、流和结果。
void RaivasDet5_Destroy() { apiModel().dispose(); }
/// 验证模型路径，再加载并校验可选类别和批次。
int RaivasDet5_Init(const char* path, int classes, int batch) {
    if (!path || !*path) return apiModel().retired("engine 路径不能为空");
    return apiModel().init(path, classes, batch);
}
/// 更新默认实例的切片参数及任务阈值。
int RaivasDet5_setPara(int size, int overlap, double threshold) { return apiModel().setPara(size, overlap, threshold); }
/// 校验输入和输出路径，然后执行 ONNX 到 engine 的转换。
int RaivasDet5_onnxToEngine(const char* source, const char* destination, bool fp16) {
    if (!source || !*source || !destination || !*destination) return apiModel().retired("ONNX 和 engine 路径不能为空");
    return apiModel().convert(source, destination, fp16);
}
/// 旧 WTS 导出只返回迁移说明，避免继续使用手写网络结构。
int RaivasDet5_wtsToEngine(const char*, const char*, const char*, int) {
    return apiModel().retired("WTS 转换已废弃，请使用 RaivasDet5_onnxToEngine");
}
/// 执行整图推理并写入结果数量。
int RaivasDet5_Detect(cv::Mat& frame, int& len) { return apiModel().run(frame, len); }
/// 读取默认实例缓存的坐标、分数、类别及可选掩码。
int RaivasDet5_GetDefectDetResult(int index, int* classId, float* score, float* x, float* y, float* w, float* h) { return apiModel().result(index, classId, score, x, y, w, h); }
/// 校验输出指针后读取默认实例自动参数。
int RaivasDet5_GetModelInfo(int* batch, int* width, int* height, int* classes) {
    if (!batch || !width || !height || !classes) return apiModel().retired("模型信息输出指针不能为空");
    return apiModel().modelInfo(*batch, *width, *height, *classes);
}
/// 将锁内复制的错误存入线程局部变量，避免返回内部字符串悬空指针。
const char* RaivasDet5_GetLastError() {
    thread_local std::string message;
    message = apiModel().error();
    return message.c_str();
}
