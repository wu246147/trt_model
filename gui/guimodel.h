#pragma once
#include <QString>
#include <memory>
#include <opencv2/core.hpp>

/// 界面支持的九个独立测试入口；枚举顺序与模型选择框一致。
enum class GuiTask { Yolo5Det, Yolo5Seg, Yolo11Det, Yolo11Seg, Yolo26Det, Yolo26Seg,
                     AnomalyDino, Dinomaly, EfficientAd };

/// 从模型自动获取的输入和类别信息，不允许界面手工覆盖。
struct GuiModelInfo {
    int batch = 0, width = 0, height = 0, classes = 0;
    QString details;
};

/// 界面显示的推理快照，矩阵拥有独立数据，不引用后台模型缓冲区。
struct GuiPrediction {
    cv::Mat overlay, mask, heatmap;
    int count = 0;
    QString details;
};

/// 将九种算法统一为界面操作，具体 TensorRT 和模型头文件仅在实现中引用。
class GuiModel {
public:
    /// 销毁当前适配器及其独立模型资源。
    virtual ~GuiModel() = default;
    /// 从 ONNX 构建引擎；失败抛出带中文上下文的异常供界面显示。
    virtual void build(const QString& onnxPath, const QString& enginePath, bool fp16) = 0;
    /// 初始化引擎，返回自动识别的输入尺寸、批次和类别信息。
    virtual GuiModelInfo load(const QString& enginePath) = 0;
    /// 使用输入图像和阈值推理，返回绘制结果；tiles 控制是否按 tileSize 和 overlap 切片。
    virtual GuiPrediction run(const cv::Mat& image, double threshold, bool tiles,
                              int tileSize, int overlap) = 0;
};

/// 创建指定测试任务的适配器；返回独立实例供后台任务持有。
std::shared_ptr<GuiModel> createGuiModel(GuiTask task);
/// 返回指定任务的中文显示名称。
QString guiTaskName(GuiTask task);
/// 判断任务是否为异常检测，用于调整阈值和可视化控件。
bool isAnomalyTask(GuiTask task);
/// 判断任务是否输出实例掩码，用于显示掩码概率阈值说明。
bool isSegmentationTask(GuiTask task);
