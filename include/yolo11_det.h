#pragma once
#include <opencv2/core.hpp>
#include <string>
#if defined(_WIN32)
# if defined(TRT_MODEL_EXPORTS)
#  define RAIVAS_DET11_API __declspec(dllexport)
# else
#  define RAIVAS_DET11_API __declspec(dllimport)
# endif
#else
# define RAIVAS_DET11_API __attribute__((visibility("default")))
#endif

// 返回码：0 成功；-1 文件/TRT/CUDA 错误；-2 参数或模型不匹配；-3 尚未初始化。
namespace RaivasDefectDet11 {
/// 将 ONNX 直接转换为 TensorRT 8.6 engine；fp16 仅控制内部计算，输入输出仍为 FP32，失败返回负值。
RAIVAS_DET11_API int onnxToEngine(std::string onnxPath, std::string enginePath, bool fp16 = true);
/// 已废弃的 WTS 构建入口，保留源码迁移兼容；始终返回 -2，请改用 onnxToEngine。
RAIVAS_DET11_API int wtsToEngine(std::string wtsPath, std::string enginePath, std::string subType, int classCount);
/// 每个对象独立管理模型与结果，内部锁保证同实例调用串行；不同实例可以并行。
class RAIVAS_DET11_API RaivasDefectDet11 {
public:
    /// 创建尚未初始化的 目标检测 对象，不在构造阶段访问 CUDA。
    RaivasDefectDet11();
    /// 释放本实例的 TensorRT、CUDA 和结果资源。
    ~RaivasDefectDet11();
    /// 禁止拷贝，避免不同对象共享底层 CUDA 所有权。
    RaivasDefectDet11(const RaivasDefectDet11&) = delete;
    /// 禁止拷贝赋值，避免重复释放原生资源。
    RaivasDefectDet11& operator=(const RaivasDefectDet11&) = delete;
    /// 配置切片边长、重叠像素和检测置信度阈值，阈值须在 [0,1]。
    int setPara(int cutSize, int overlappingSize, double threshold);
    /// 从 engine 自动读取尺寸、类别和批次；classSize/batchSize 非正数自动读取，正数必须匹配。
    int initModel(std::string modelPath, int classSize = 0, int batchSize = 0);
    /// 转换 ONNX 为 engine，自动读取模型结构及输入输出；失败通过 getLastError 返回原因。
    int onnxToEngine(std::string onnxPath, std::string enginePath, bool fp16 = true);
    /// 释放本实例资源和推理结果，允许重复调用并重新初始化。
    int disposeModel();
    /// 推理一张非空 8 位灰度/BGR/BGRA 整图，len 写入实例数；失败 len 为零。
    int runModel(cv::Mat& inputImage, int& len);
    /// 按配置切片并组批推理，len 写入实例数，结果映射回整图；重叠区域可能有重复目标。
    int runModelCutImage(cv::Mat& inputImage, int& len);
    /// 兼容 YOLO5 的原切片调用名，其行为与 runModelCutImage 相同。
    int runCutModel(cv::Mat& inputImage, int& len);
    /// 读取 [0,len) 内的原图 x/y/width/height，输出指针不能为空。
    int getDefectDetResult(int index, int* classId, float* score, float* x, float* y, float* w, float* h);
    /// 读取自动识别的批次、输入宽高和类别；未初始化返回 -3。
    int getModelInfo(int& batch, int& width, int& height, int& classes) const;
    /// 返回最近错误详情的独立副本；同实例并发读取安全。
    std::string getLastError() const;
    /// 已废弃的 WTS 构建入口，始终返回 -2；subType/classCount 不再用于网络构建。
    int wtsToEngine(std::string wtsPath, std::string enginePath, std::string subType, int classCount);
    /// 兼容旧 YOLO11 三参数转换入口，已废弃并返回 -2，请改用 onnxToEngine。
    int wtsToEngine(std::string wtsPath, std::string enginePath, std::string subType);
    /// 架构配置已由 ONNX 替代，此旧入口返回 -2 并说明迁移方式。
    int initArch(std::string paraPath);
private:
    struct Impl;
    Impl* impl_;
};
}
extern "C" {
/// 释放默认实例的资源，不影响独立 C++ 实例或其他 DLL。
RAIVAS_DET11_API void RaivasDet11_Destroy();
/// 加载 engine；classSize/batchSize 默认自动读取，正数须与模型一致。
RAIVAS_DET11_API int RaivasDet11_Init(const char* enginePath, int classSize = 0, int batchSize = 0);
/// 设置默认实例的切片和置信度阈值，参数非法时返回 -2。
RAIVAS_DET11_API int RaivasDet11_setPara(int cutSize, int overlappingSize, double threshold);
/// ONNX 直接构建 engine；路径不能为空，返回 0 成功或负值错误。
RAIVAS_DET11_API int RaivasDet11_onnxToEngine(const char* onnxPath, const char* enginePath, bool fp16 = true);
/// 兼容旧符号；WTS 已停止支持并返回 -2，可通过 GetLastError 获取迁移说明。
RAIVAS_DET11_API int RaivasDet11_wtsToEngine(const char* wtsPath, const char* enginePath, const char* subType, int classCount);
/// 推理整图并覆盖默认实例的上次结果，len 返回有效实例数。
RAIVAS_DET11_API int RaivasDet11_Detect(cv::Mat& frame, int& len);
/// 读取默认实例的原图坐标结果，无效下标或空指针返回 -2。
RAIVAS_DET11_API int RaivasDet11_GetDefectDetResult(int index, int* classId, float* score, float* x, float* y, float* w, float* h);
/// 读取默认实例自动参数，输出指针不能为空。
RAIVAS_DET11_API int RaivasDet11_GetModelInfo(int* batch, int* width, int* height, int* classes);
/// 返回当前线程持有的错误字符串；有效期至本线程下一次调用此函数。
RAIVAS_DET11_API const char* RaivasDet11_GetLastError();
/// 架构文件不再使用，保留符号并返回 -2；改由 ONNX 自动提供参数。
RAIVAS_DET11_API int RaivasDet11_InitArch(const char* paraPath);
}
