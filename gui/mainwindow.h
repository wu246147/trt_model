#pragma once
#include <QMainWindow>
#include <memory>

/// 统一 TensorRT 测试窗口，采用左侧模型控制、中间图像和右侧目录的布局。
class MainWindow final : public QMainWindow {
    Q_OBJECT
public:
    /// 创建九种任务的独立状态；rememberSettings 控制是否读取和保存路径。
    explicit MainWindow(QWidget* parent = nullptr, bool rememberSettings = true);
    /// 等待后台工作结束后释放模型、图像和窗口资源。
    ~MainWindow() override;
    /// 用指定任务、引擎和图像执行集成检查；tiles、tileSize、overlap 控制切片，保存截图/结果，零表示成功。
    int smokeTest(int taskIndex, const QString& engine, const QString& image,
                  const QString& outputDirectory, bool tiles = false, int tileSize = 640, int overlap = 64);
protected:
    /// 后台转换或推理期间拒绝关闭，防止销毁仍被工作线程使用的状态。
    void closeEvent(QCloseEvent* event) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};


void MainWindow::Impl::on_inferButton_clicked()
{

}


