#pragma once
#include <memory>
#include <string>
#include <opencv2/core.hpp>

#if defined(_WIN32)
# if defined(ANOMALY_TRT_EXPORTS)
#  define ANOMALY_TRT_API __declspec(dllexport)
# else
#  define ANOMALY_TRT_API __declspec(dllimport)
# endif
#else
# define ANOMALY_TRT_API __attribute__((visibility("default")))
#endif

/// 三个已校准异常检测模型；加载时检查引擎保存的模型类型。
enum class AnomalyKind { AnomalyDINO, Dinomaly, EfficientAD };

/// 自动从模型元数据获得的尺寸、类别与校准阈值。
struct AnomalyModelInfo {
    std::string modelType, normalClass, anomalyClass;
    cv::Size resizeSize, networkSize;
    float imageThreshold = 0.5f, pixelThreshold = 0.5f, defaultPixelThreshold = 0.5f;
    bool preprocessingEmbedded = false;
    int batchSize = 0;
};

/// 单张异常检测结果；图像均恢复为原图尺寸，mask为0/255，异常图为0..1。
struct AnomalyResult {
    float score = 0;
    int classId = -1;
    std::string className;
    cv::Mat anomalyMap, mask, heatmap, overlay;
    double elapsedMs = 0;
};

/// 三个异常模型共用的 TensorRT 执行接口；同一实例的调用串行化，错误通过 lastError 查询。
class ANOMALY_TRT_API AnomalyTRT {
public:
    /// 创建指定模型类型的空实例，尚不分配GPU资源。
    explicit AnomalyTRT(AnomalyKind kind);
    /// 释放本实例模型、执行上下文和缓存，不重置CUDA设备。
    virtual ~AnomalyTRT();
    /// 禁止复制执行器，防止共享GPU上下文与结果缓存。
    AnomalyTRT(const AnomalyTRT&) = delete;
    /// 禁止复制赋值，保证每个模型实例独占资源与互斥状态。
    AnomalyTRT& operator=(const AnomalyTRT&) = delete;
    /// 构建已校准引擎；默认FP32，fp16允许半精度；Dinomaly固定FP32并记录实际精度，失败返回-1及具体错误。
    int onnxToEngine(std::string onnxPath, std::string enginePath, bool fp16 = false);
    /// 加载引擎并校验类型、输出与元数据，自动读取输入尺寸和类别；失败清空旧模型。
    int initModel(std::string enginePath);
    /// 设置归一化图片/像素阈值，范围[0,1]；-1表示使用模型校准值，失败返回-2。
    int setPara(double imageThreshold = -1, double pixelThreshold = -1);
    /// 设置方形切片尺寸及重叠像素，要求cutSize>0且0<=overlap<cutSize；失败返回-2且保留原配置。
    int setCutPara(int cutSize, int overlap);
    /// 按YOLO风格同时设置切片尺寸、重叠与像素阈值；图片阈值保持当前值，pixelThreshold=-1使用校准值。
    int setPara(int cutSize, int overlappingSize, double pixelThreshold);
    /// 推理8位灰度、BGR或BGRA整图；成功len=1，失败len=0；每次调用替换旧结果。
    int runModel(cv::Mat& inputImage, int& len);
    /// 按配置切片推理并映回整图：重叠异常图取max、mask取OR、图片得分取所有切片max，成功len=1。
    int runCutModel(cv::Mat& inputImage, int& len);
    /// runCutModel的同义入口，兼容YOLO已有切图方法命名，返回值与结果语义一致。
    int runModelCutImage(cv::Mat& inputImage, int& len);
    /// 读取最近成功推理的结果，各图像深拷贝；未有结果返回-3。
    int getResult(AnomalyResult& result) const;
    /// 返回当前模型尺寸、类别及有效阈值；未初始化时batchSize为0。
    AnomalyModelInfo modelInfo() const;
    /// 返回最近一次操作的中文错误及原始底层错误文本。
    std::string lastError() const;
    /// 释放模型及结果，可重复调用，成功返回0。
    int disposeModel();
private:
    /// 整图与切片公用的串行执行入口；cuts控制是否按切片配置划分输入，失败清空结果。
    int run(cv::Mat& inputImage, int& len, bool cuts);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// AnomalyDINO专用类，接口继承自AnomalyTRT。
class ANOMALY_TRT_API AnomalyDINOTRT final : public AnomalyTRT {
public:
    /// 创建AnomalyDINO空实例。
    AnomalyDINOTRT();
};
/// Dinomaly专用类，接口继承自AnomalyTRT。
class ANOMALY_TRT_API DinomalyTRT final : public AnomalyTRT {
public:
    /// 创建Dinomaly空实例。
    DinomalyTRT();
};
/// EfficientAD专用类，接口继承自AnomalyTRT。
class ANOMALY_TRT_API EfficientADTRT final : public AnomalyTRT {
public:
    /// 创建EfficientAD空实例。
    EfficientADTRT();
};

using RaivasAnomalyTRTHandle = void*;
extern "C" {
/// 创建指定模型的独立句柄，kind为AnomalyKind整数值；无效参数或分配失败返回空。
ANOMALY_TRT_API RaivasAnomalyTRTHandle RaivasAnomalyTRT_Create(int kind);
/// 释放句柄及其GPU资源；空句柄可重复传入，已释放的非空句柄不可复用。
ANOMALY_TRT_API void RaivasAnomalyTRT_Destroy(RaivasAnomalyTRTHandle handle);
/// 将ONNX转为engine；路径为UTF-8，成功0，参数无效-2；默认FP32，Dinomaly始终使用FP32精度保护。
ANOMALY_TRT_API int RaivasAnomalyTRT_onnxToEngine(RaivasAnomalyTRTHandle handle, const char* onnxPath, const char* enginePath, bool fp16 = false);
/// 加载引擎并自动读取尺寸及校准信息，成功返回0。
ANOMALY_TRT_API int RaivasAnomalyTRT_Init(RaivasAnomalyTRTHandle handle, const char* enginePath);
/// 设置图片和像素阈值；-1恢复模型默认值，范围[0,1]。
ANOMALY_TRT_API int RaivasAnomalyTRT_setPara(RaivasAnomalyTRTHandle handle, double imageThreshold, double pixelThreshold);
/// 设置切片尺寸和重叠像素，不改变图片及像素阈值。
ANOMALY_TRT_API int RaivasAnomalyTRT_setCutPara(RaivasAnomalyTRTHandle handle, int cutSize, int overlap);
/// 推理图像并返回深拷贝结果；frame/result不可为空，成功0。
ANOMALY_TRT_API int RaivasAnomalyTRT_Detect(RaivasAnomalyTRTHandle handle, cv::Mat* frame, AnomalyResult* result);
/// 按配置切片推理并返回整图异常图/掩码与最大切片分数，frame/result不可为空。
ANOMALY_TRT_API int RaivasAnomalyTRT_DetectCut(RaivasAnomalyTRTHandle handle, cv::Mat* frame, AnomalyResult* result);
/// 查询引擎尺寸和阈值；info不可为空。
ANOMALY_TRT_API int RaivasAnomalyTRT_GetModelInfo(RaivasAnomalyTRTHandle handle, AnomalyModelInfo* info);
/// 复制UTF-8错误文本并以零结尾，返回所需缓冲区大小（含零）；无效参数返回-2。
ANOMALY_TRT_API int RaivasAnomalyTRT_GetLastError(RaivasAnomalyTRTHandle handle, char* buffer, int capacity);
}
