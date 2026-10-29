#include "mainwindow.h"
#include "guimodel.h"
#include "imageview.h"
#include "ui_mainwindow.h"
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTimer>
#include <QToolBar>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <functional>
#include <stdexcept>
#include <vector>

namespace {
/// 从 Unicode 路径读取图片，避免 OpenCV 窄字符文件接口在 Windows 上丢失中文路径。
cv::Mat readImage(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) throw std::runtime_error("无法打开图像文件：" + path.toStdString());
    const QByteArray bytes = file.readAll();
    const std::vector<unsigned char> data(bytes.begin(), bytes.end());
    cv::Mat image = cv::imdecode(data, cv::IMREAD_UNCHANGED);
    if (image.empty()) throw std::runtime_error("图像解码失败：" + path.toStdString());
    if (image.depth() != CV_8U || (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
        throw std::runtime_error("测试界面支持 8 位灰度、BGR 或 BGRA 图像");
    return image;
}

/// 通过 Qt 文件接口写入已编码图片，支持 Unicode 路径并验证实际写入长度。
void writeImage(const QString& path, const cv::Mat& image) {
    if (image.empty()) throw std::runtime_error("当前没有可保存的图像");
    std::vector<unsigned char> data;
    const QString extension = "." + QFileInfo(path).suffix().toLower();
    if (!cv::imencode(extension.toStdString(), image, data)) throw std::runtime_error("图像编码失败");
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(reinterpret_cast<const char*>(data.data()),
        static_cast<qint64>(data.size())) != static_cast<qint64>(data.size()))
        throw std::runtime_error("图像保存失败：" + path.toStdString());
}

/// 将 OpenCV 数据复制为 Qt 图像，避免显示对象引用已释放的临时矩阵。
QImage qtImage(const cv::Mat& image) {
    if (image.empty()) return {};
    if (image.channels() == 1)
        return QImage(image.data, image.cols, image.rows, static_cast<int>(image.step), QImage::Format_Grayscale8).copy();
    cv::Mat rgb;
    cv::cvtColor(image, rgb, image.channels() == 4 ? cv::COLOR_BGRA2RGB : cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
}

enum class Operation { Build, Load, Infer };
/// 后台工作只返回数据，不访问任何 Qt 控件；主线程统一更新界面。
struct WorkResult {
    bool success = false;
    QString error;
    GuiModelInfo info;
    GuiPrediction prediction;
    double elapsedMs = 0;
};
}

struct MainWindow::Impl {
    /// 每个测试项保存独立模型、路径、输入及结果，切换任务不会串用另一个模型的数据。
    struct TaskState {
        std::shared_ptr<GuiModel> model;
        QString onnx, engine, imagePath, loadedPath;
        GuiModelInfo info;
        cv::Mat image;
        GuiPrediction prediction;
        bool initialized = false;
        bool allowFp16 = true;
        double threshold = 0.5;
    };
    MainWindow& window;
    const bool remember;
    Ui::MainWindow ui;
    std::array<TaskState, 9> tasks;
    int current = -1, workerTask = 0;
    Operation operation = Operation::Load;
    bool busy = false, repeating = false;
    QWidget* controls = nullptr;
    QComboBox *taskChoice = nullptr, *displayChoice = nullptr;
    QLineEdit *onnxPath = nullptr, *enginePath = nullptr, *imagePath = nullptr;
    QCheckBox *fp16 = nullptr, *tiles = nullptr, *defaultPixelThreshold = nullptr;
    QDoubleSpinBox* threshold = nullptr;
    QSpinBox *tileSize = nullptr, *overlap = nullptr;
    QLabel *modelInfo = nullptr, *thresholdLabel = nullptr, *resultInfo = nullptr, *directoryLabel = nullptr;
    QPushButton* continuous = nullptr;
    QListWidget* files = nullptr;
    QPlainTextEdit* log = nullptr;
    ImageView* view = nullptr;
    QFutureWatcher<WorkResult> watcher;
    QTimer timer;
    QString directory;
    cv::Mat displayed;

    /// 创建控件并连接主线程事件；不在启动时加载任何 GPU 模型。
    Impl(MainWindow& owner, bool rememberSettings) : window(owner), remember(rememberSettings) {
        setupUi();
        restoreSettings();
        showTask(0);
        QObject::connect(&watcher, &QFutureWatcher<WorkResult>::finished, &window, [this] { finishWork(); });
        QObject::connect(&timer, &QTimer::timeout, &window, [this] {
            if (repeating && !busy) infer();
        });
        timer.setInterval(50);
    }

    /// 等待不访问 UI 的后台工作完成，随后保存可恢复的界面路径。
    ~Impl() {
        timer.stop();
        watcher.waitForFinished();
        saveSettings();
    }

    /// 打开 .ui 内路径控件对应的文件对话框，输出引擎使用保存对话框。
    void browsePath(QLineEdit* field, const QString& filter, bool save = false) {
        if (busy) return;
        const QString path = save
            ? QFileDialog::getSaveFileName(&window, "选择输出或已有引擎", field->text(), filter,
                nullptr, QFileDialog::DontConfirmOverwrite)
            : QFileDialog::getOpenFileName(&window, "选择文件", field->text(), filter);
        if (!path.isEmpty()) field->setText(path);
    }

    /// 从 Designer 文件建立全部控件和布局，代码仅绑定成员及模型操作事件。
    void setupUi() {
        ui.setupUi(&window);
        controls = ui.controls;
        taskChoice = ui.taskChoice;
        displayChoice = ui.displayChoice;
        onnxPath = ui.onnxPath;
        enginePath = ui.enginePath;
        imagePath = ui.imagePath;
        fp16 = ui.fp16CheckBox;
        tiles = ui.tilesCheckBox;
        defaultPixelThreshold = ui.defaultPixelThresholdCheckBox;
        threshold = ui.thresholdSpinBox;
        tileSize = ui.tileSizeSpinBox;
        overlap = ui.overlapSpinBox;
        modelInfo = ui.modelInfo;
        thresholdLabel = ui.thresholdLabel;
        resultInfo = ui.resultInfo;
        directoryLabel = ui.directoryLabel;
        continuous = ui.continuousButton;
        files = ui.imageFiles;
        log = ui.operationLog;
        view = ui.imageView;
        ui.columnsSplitter->setSizes({350, 800, 220});
        ui.columnsSplitter->setStretchFactor(1, 1);
        ui.verticalSplitter->setSizes({710, 140});
        ui.verticalSplitter->setStretchFactor(0, 1);
        window.statusBar()->showMessage("就绪");

        QObject::connect(ui.actionOpenImage, &QAction::triggered, &window, [this] { if (!busy) chooseImage(); });
        QObject::connect(ui.actionOpenDirectory, &QAction::triggered, &window, [this] { if (!busy) chooseDirectory(); });
        QObject::connect(ui.actionSaveImage, &QAction::triggered, &window, [this] { saveCurrentImage(); });
        QObject::connect(ui.actionFitImage, &QAction::triggered, &window, [this] { view->fitImage(); });
        QObject::connect(ui.actionStopContinuous, &QAction::triggered, &window, [this] { stopContinuous(); });
        QObject::connect(ui.browseOnnxButton, &QPushButton::clicked, &window,
            [this] { browsePath(onnxPath, "ONNX 模型 (*.onnx)"); });
        QObject::connect(ui.browseEngineButton, &QPushButton::clicked, &window,
            [this] { browsePath(enginePath, "TensorRT 引擎 (*.engine)", true); });
        QObject::connect(ui.browseImageButton, &QPushButton::clicked, &window,
            [this] { browsePath(imagePath, "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"); });
        QObject::connect(taskChoice, qOverload<int>(&QComboBox::currentIndexChanged), &window,
            [this](int index) { if (!busy) showTask(index); });
        QObject::connect(ui.convertButton, &QPushButton::clicked, &window, [this] { convertModel(); });
        QObject::connect(ui.initializeButton, &QPushButton::clicked, &window, [this] { initialize(); });
        QObject::connect(ui.releaseModelButton, &QPushButton::clicked, &window, [this] { releaseModel(); });
        QObject::connect(ui.loadImageButton, &QPushButton::clicked, &window,
            [this] { loadCurrentImage(imagePath->text().trimmed(), true); });
        QObject::connect(ui.inferButton, &QPushButton::clicked, &window, [this] { infer(); });
        QObject::connect(continuous, &QPushButton::clicked, &window, [this] { toggleContinuous(); });
        QObject::connect(ui.directoryButton, &QPushButton::clicked, &window, [this] { if (!busy) chooseDirectory(); });
        QObject::connect(displayChoice, qOverload<int>(&QComboBox::currentIndexChanged), &window,
            [this](int) { display(); });
        QObject::connect(defaultPixelThreshold, &QCheckBox::toggled, &window,
            [this](bool checked) { threshold->setEnabled(!isAnomalyTask(static_cast<GuiTask>(current)) || !checked); });
        QObject::connect(files, &QListWidget::currentItemChanged, &window,
            [this](QListWidgetItem* item, QListWidgetItem*) {
                if (item && !busy) loadCurrentImage(item->data(Qt::UserRole).toString(), true);
            });
        // 切换模型后列表的当前项可能没变，单击同一项也应为新任务加载图片。
        QObject::connect(files, &QListWidget::itemClicked, &window,
            [this](QListWidgetItem* item) {
                if (!item || busy) return;
                const QString path = item->data(Qt::UserRole).toString();
                if (tasks[current].image.empty() || tasks[current].loadedPath != path) loadCurrentImage(path, true);
            });
    }
    /// 向只读日志追加时间和操作结果，同时在状态栏显示最近消息。
    void report(const QString& text, bool error = false) {
        log->appendPlainText(QDateTime::currentDateTime().toString("HH:mm:ss ") + (error ? "[错误] " : "") + text);
        window.statusBar()->showMessage(text);
    }

    /// 保存当前控件到任务状态，所有模型的路径、阈值和转换精度相互独立。
    void captureFields() {
        if (current < 0 || current >= static_cast<int>(tasks.size())) return;
        auto& state = tasks[current];
        state.onnx = onnxPath->text().trimmed();
        state.engine = enginePath->text().trimmed();
        state.imagePath = imagePath->text().trimmed();
        state.threshold = isAnomalyTask(static_cast<GuiTask>(current)) && defaultPixelThreshold->isChecked()
            ? -1.0 : threshold->value();
        state.allowFp16 = fp16->isChecked();
    }

    /// 切换任务并恢复模型信息、图片和结果；Dinomaly 固定 FP32，避免半精度溢出及临界缺陷漏检。
    void showTask(int index) {
        if (index < 0 || index >= static_cast<int>(tasks.size())) return;
        stopContinuous();
        captureFields();
        current = index;
        auto& state = tasks[current];
        onnxPath->setText(state.onnx);
        enginePath->setText(state.engine);
        imagePath->setText(state.imagePath);
        const auto task = static_cast<GuiTask>(current);
        const bool forceFp32 = task == GuiTask::Dinomaly;
        if (forceFp32) state.allowFp16 = false;
        fp16->setChecked(state.allowFp16);
        fp16->setEnabled(!forceFp32);
        fp16->setText(forceFp32 ? "Dinomaly 固定 FP32（精度保护）" : "允许 FP16 内部计算");
        fp16->setToolTip(forceFp32
            ? "本模型半精度会发生注意力溢出；局部精度保护仍使临界缺陷误判，当前固定使用 FP32。旧 FP16 engine 需要重新转换。"
            : "输入输出保持 FP32；部分模型的 FP16 精度需要与 FP32 结果比较确认。");
        const bool anomaly = isAnomalyTask(task);
        defaultPixelThreshold->setVisible(anomaly);
        defaultPixelThreshold->setChecked(state.threshold < 0);
        threshold->setValue(state.threshold < 0 ? 0.5 : state.threshold);
        threshold->setEnabled(!anomaly || !defaultPixelThreshold->isChecked());
        thresholdLabel->setText(anomaly ? "像素阈值" : isSegmentationTask(task) ? "掩码阈值" : "置信度阈值");
        threshold->setToolTip(anomaly ? "仅覆盖像素掩码阈值；整图判定使用模型校准阈值。"
            : isSegmentationTask(task) ? "实例掩码概率阈值；实例检测置信度保持兼容值 0.5。" : "目标检测的置信度阈值。");
        modelInfo->setText(state.initialized ? state.info.details : "尚未初始化；模型参数将自动读取");
        resultInfo->setText(state.prediction.details.isEmpty() ? "加载模型和图片后开始测试" : state.prediction.details);
        display();
    }

    /// 确保当前任务有独立适配器，首次操作才创建实际对象。
    std::shared_ptr<GuiModel> model() {
        auto& state = tasks[current];
        if (!state.model) state.model = createGuiModel(static_cast<GuiTask>(current));
        return state.model;
    }

    /// 后台工作期间禁用会改变模型和输入的控件，避免并发状态交叉。
    void setBusy(bool value) {
        busy = value;
        controls->setEnabled(!value);
        files->setEnabled(!value);
    }

    /// 在工作线程执行操作，捕获异常并将完整错误带回主线程；不跨线程访问控件。
    void startWork(Operation kind, std::function<WorkResult()> work) {
        if (busy) return;
        operation = kind;
        workerTask = current;
        setBusy(true);
        window.statusBar()->showMessage(kind == Operation::Build ? "正在转换 ONNX，较大模型可能需要数分钟…"
            : kind == Operation::Load ? "正在加载模型…" : "正在推理…");
        watcher.setFuture(QtConcurrent::run([work = std::move(work)] {
            QElapsedTimer elapsed;
            elapsed.start();
            WorkResult result;
            try {
                result = work();
                result.success = true;
            } catch (const std::exception& error) {
                result.error = QString::fromUtf8(error.what());
            } catch (...) {
                result.error = "模型操作遇到未知异常";
            }
            result.elapsedMs = elapsed.nsecsElapsed() / 1000000.0;
            return result;
        }));
    }

    /// 主线程消费后台结果；失败时停止连续推理并清除本次失效结果。
    void finishWork() {
        const WorkResult result = watcher.result();
        setBusy(false);
        auto& state = tasks[workerTask];
        if (!result.success) {
            if (operation == Operation::Load) state.initialized = false;
            if (operation == Operation::Infer) state.prediction = {};
            stopContinuous();
            report(result.error, true);
            if (operation == Operation::Load) modelInfo->setText("初始化失败");
            if (operation == Operation::Infer) { resultInfo->setText("推理失败"); display(); }
            return;
        }
        const QString elapsed = QString::number(result.elapsedMs, 'f', 2);
        if (operation == Operation::Build) {
            // 内存中的旧引擎不会随磁盘文件更新，必须显式重新初始化后才能推理。
            state.initialized = false;
            state.prediction = {};
            modelInfo->setText("转换完成，请初始化新引擎");
            resultInfo->setText("等待初始化新引擎");
            display();
            report(QString("ONNX 转换完成，耗时 %1 ms；输出 %2").arg(elapsed, state.engine));
        } else if (operation == Operation::Load) {
            state.initialized = true;
            state.info = result.info;
            modelInfo->setText(state.info.details);
            report(QString("%1 初始化完成，耗时 %2 ms；%3")
                .arg(guiTaskName(static_cast<GuiTask>(workerTask)), elapsed, state.info.details));
        } else {
            state.prediction = result.prediction;
            state.prediction.details += QString("；总耗时 %1 ms").arg(elapsed);
            resultInfo->setText(state.prediction.details);
            display();
            report(state.prediction.details);
        }
    }

    /// 集成检查专用：运行事件循环直到当前后台操作完成或超时，允许真实按钮回调更新界面。
    bool waitForWork(int timeoutMs) {
        if (!busy) return watcher.isFinished() && watcher.result().success;
        QEventLoop loop;
        QTimer timeout;
        timeout.setSingleShot(true);
        QObject::connect(&watcher, &QFutureWatcher<WorkResult>::finished, &loop, &QEventLoop::quit);
        QObject::connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
        timeout.start(timeoutMs);
        loop.exec();
        return !busy && watcher.result().success;
    }

    /// 检查文件路径并启动 ONNX 转换；输出名为空时使用 ONNX 同名 engine 文件。
    void convertModel() {
        if (busy) return;
        stopContinuous();
        captureFields();
        auto& state = tasks[current];
        if (!QFileInfo::exists(state.onnx)) { report("请先选择存在的 ONNX 模型", true); return; }
        if (state.engine.isEmpty()) {
            const QFileInfo source(state.onnx);
            state.engine = source.dir().filePath(source.completeBaseName() + ".engine");
            enginePath->setText(state.engine);
        }
        const auto instance = model();
        const QString onnx = state.onnx, engine = state.engine;
        const bool half = fp16->isChecked();
        report("开始转换 " + guiTaskName(static_cast<GuiTask>(current)));
        startWork(Operation::Build, [instance, onnx, engine, half] {
            instance->build(onnx, engine, half);
            return WorkResult{};
        });
    }

    /// 使用模型当前 engine 路径初始化；结果成功之前禁止推理此模型。
    void initialize() {
        if (busy) return;
        stopContinuous();
        captureFields();
        auto& state = tasks[current];
        if (state.engine.isEmpty()) { report("请先选择 engine 文件", true); return; }
        state.initialized = false;
        state.prediction = {};
        const auto instance = model();
        const QString path = state.engine;
        startWork(Operation::Load, [instance, path] {
            WorkResult result;
            result.info = instance->load(path);
            return result;
        });
    }

    /// 释放当前模型及其显存，保留文件路径和输入图片，以便连续测试多个较大的模型。
    void releaseModel() {
        if (busy) return;
        stopContinuous();
        auto& state = tasks[current];
        state.model.reset();
        state.initialized = false;
        state.info = {};
        state.prediction = {};
        modelInfo->setText("模型已释放；可重新初始化");
        resultInfo->setText("当前模型资源已释放");
        display();
        report(guiTaskName(static_cast<GuiTask>(current)) + " 资源已释放");
    }

    /// 读取图片并清除旧结果；自动推理仅在当前任务已初始化时执行。
    bool loadCurrentImage(const QString& path, bool autoInfer) {
        if (busy) return false;
        stopContinuous();
        try {
            cv::Mat image = readImage(path);
            auto& state = tasks[current];
            state.image = std::move(image);
            state.imagePath = path;
            state.loadedPath = path;
            state.prediction = {};
            imagePath->setText(path);
            resultInfo->setText(QString("原图 %1 × %2").arg(state.image.cols).arg(state.image.rows));
            display();
            report("已加载 " + path);
            if (autoInfer && state.initialized) infer();
            return true;
        } catch (const std::exception& error) {
            report(QString::fromUtf8(error.what()), true);
            return false;
        }
    }

    /// 打开图像文件选择框，并通过统一的 Unicode 文件读取流程加载。
    void chooseImage() {
        const QString path = QFileDialog::getOpenFileName(&window, "打开图片", imagePath->text(),
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)");
        if (!path.isEmpty()) loadCurrentImage(path, true);
    }

    /// 将目录内支持的图像列在右侧，实际加载由列表选中事件触发。
    void chooseDirectory() {
        const QString path = QFileDialog::getExistingDirectory(&window, "打开图片目录", directory);
        if (!path.isEmpty()) populateDirectory(path);
    }

    /// 按文件名排序填充目录列表，存储完整路径以避免目录切换导致误读。
    void populateDirectory(const QString& path) {
        directory = path;
        directoryLabel->setText(path);
        const QSignalBlocker block(files);
        files->clear();
        const QDir folder(path);
        const auto entries = folder.entryInfoList({"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"},
            QDir::Files | QDir::Readable, QDir::Name);
        for (const auto& entry : entries) {
            auto* item = new QListWidgetItem(entry.fileName(), files);
            item->setData(Qt::UserRole, entry.absoluteFilePath());
            item->setToolTip(entry.absoluteFilePath());
        }
        report(QString("目录中有 %1 张图片").arg(entries.size()));
    }

    /// 验证当前图像与切片参数后，在后台执行推理并返回独立结果快照。
    void infer() {
        if (busy) return;
        captureFields();
        auto& state = tasks[current];
        if (!state.initialized) { stopContinuous(); report("请先初始化当前模型", true); return; }
        if (state.image.empty()) { stopContinuous(); report("请先打开图像", true); return; }
        const bool useTiles = tiles->isChecked();
        const int size = tileSize->value(), padding = overlap->value();
        if (useTiles && padding >= size) { stopContinuous(); report("重叠像素必须小于切片尺寸", true); return; }
        const auto instance = state.model;
        // Mat 采用引用计数持有当前输入；忙碌期间禁止换图，模型只读取此图像。
        const cv::Mat image = state.image;
        const double value = state.threshold;
        startWork(Operation::Infer, [instance, image, value, useTiles, size, padding] {
            WorkResult result;
            result.prediction = instance->run(image, value, useTiles, size, padding);
            return result;
        });
    }

    /// 开始或停止对当前图像的连续推理；只允许一个后台操作同时运行。
    void toggleContinuous() {
        if (repeating) { stopContinuous(); return; }
        if (!tasks[current].initialized || tasks[current].image.empty()) {
            report("连续检测需要先初始化模型并打开图片", true);
            return;
        }
        repeating = true;
        continuous->setText("停止连续检测");
        timer.start();
        infer();
    }

    /// 停止发起下一次推理，已经提交的当前操作会安全完成。
    void stopContinuous() {
        repeating = false;
        timer.stop();
        if (continuous) continuous->setText("连续检测");
    }

    /// 根据可视化选择显示原图、叠加结果、掩码或热力图；缺少结果时显示原图。
    void display() {
        const auto& state = tasks[current];
        const int choice = displayChoice->currentIndex();
        const cv::Mat* image = &state.image;
        if (choice == 0 && !state.prediction.overlay.empty()) image = &state.prediction.overlay;
        if (choice == 2 && !state.prediction.mask.empty()) image = &state.prediction.mask;
        if (choice == 3 && !state.prediction.heatmap.empty()) image = &state.prediction.heatmap;
        displayed = *image;
        view->setImage(qtImage(displayed));
    }

    /// 保存当前实际显示的图像，图像编码和文件写入错误均写入操作日志。
    void saveCurrentImage() {
        if (displayed.empty()) { report("当前没有可保存的图像", true); return; }
        QString path = QFileDialog::getSaveFileName(&window, "保存当前图像", "result.png", "PNG 图像 (*.png);;JPEG 图像 (*.jpg)");
        if (path.isEmpty()) return;
        if (QFileInfo(path).suffix().isEmpty()) path += ".png";
        try { writeImage(path, displayed); report("已保存 " + path); }
        catch (const std::exception& error) { report(QString::fromUtf8(error.what()), true); }
    }

    /// 读取九个测试入口的路径及转换精度；异常模型首次使用 FP32 和 ONNX 像素校准阈值。
    void restoreSettings() {
        // 异常得分和校准阈值对计算精度敏感，首次使用 FP32；仍允许用户主动选择 FP16。
        for (int i = 6; i < 9; ++i) { tasks[i].threshold = -1; tasks[i].allowFp16 = false; }
        if (!remember) return;
        QSettings settings;
        for (int i = 0; i < 9; ++i) {
            settings.beginGroup(QString("task%1").arg(i));
            tasks[i].onnx = settings.value("onnx").toString();
            tasks[i].engine = settings.value("engine").toString();
            tasks[i].imagePath = settings.value("image").toString();
            tasks[i].allowFp16 = settings.value("fp16", tasks[i].allowFp16).toBool();
            settings.endGroup();
        }
        const QString folder = settings.value("directory").toString();
        if (!folder.isEmpty() && QDir(folder).exists()) populateDirectory(folder);
    }

    /// 保存文件路径、目录和各任务转换精度，不把模型尺寸或类别写成可覆盖的用户参数。
    void saveSettings() {
        if (!remember) return;
        captureFields();
        QSettings settings;
        for (int i = 0; i < 9; ++i) {
            settings.beginGroup(QString("task%1").arg(i));
            settings.setValue("onnx", tasks[i].onnx);
            settings.setValue("engine", tasks[i].engine);
            settings.setValue("image", tasks[i].imagePath);
            settings.setValue("fp16", tasks[i].allowFp16);
            settings.endGroup();
        }
        settings.setValue("directory", directory);
    }
};

/// 初始化窗口状态及九种模型的统一操作界面。
MainWindow::MainWindow(QWidget* parent, bool rememberSettings)
    : QMainWindow(parent), impl_(std::make_unique<Impl>(*this, rememberSettings)) {}

/// 由私有实现等待后台工作结束后释放模型，保证线程不访问已经销毁的数据。
MainWindow::~MainWindow() = default;

/// 后台工作执行中保持窗口存活，完成后用户可正常关闭。
void MainWindow::closeEvent(QCloseEvent* event) {
    impl_->stopContinuous();
    if (impl_->busy) {
        impl_->report("当前模型操作尚未完成，请等待操作结束后关闭窗口。");
        event->ignore();
        return;
    }
    event->accept();
}

/// 点击真实初始化、读图及整图或切片推理按钮，验证后台回调、状态隔离、图像绘制和结果导出。
int MainWindow::smokeTest(int taskIndex, const QString& engine, const QString& image,
    const QString& outputDirectory, bool useTiles, int tileSize, int overlap) {
    if (taskIndex < 0 || taskIndex >= 9 || impl_->busy) return 2;
    try {
        if (!QDir().mkpath(outputDirectory)) throw std::runtime_error("无法创建检查输出目录");
        impl_->taskChoice->setCurrentIndex(taskIndex);
        auto& state = impl_->tasks[taskIndex];
        impl_->tiles->setChecked(useTiles);
        impl_->tileSize->setValue(tileSize);
        impl_->overlap->setValue(overlap);
        impl_->enginePath->setText(engine);
        impl_->ui.initializeButton->click();
        // 初始化期间主动送达关闭事件，确认窗口会保留仍在使用的后台模型资源。
        QCloseEvent closeCheck;
        QApplication::sendEvent(this, &closeCheck);
        if (closeCheck.isAccepted()) throw std::runtime_error("后台操作期间未阻止关闭窗口");
        if (!impl_->waitForWork(300000) || !state.initialized)
            throw std::runtime_error("初始化按钮检查失败，详情见操作日志");
        impl_->imagePath->setText(image);
        impl_->ui.loadImageButton->click();
        if (!impl_->waitForWork(300000) || state.image.empty())
            throw std::runtime_error("图片加载后的自动推理检查失败，详情见操作日志");
        impl_->ui.inferButton->click();
        if (!impl_->waitForWork(300000) || state.prediction.overlay.empty())
            throw std::runtime_error("模型检测按钮检查失败，详情见操作日志");
        const QString previousEngine = impl_->enginePath->text();
        impl_->taskChoice->setCurrentIndex((taskIndex + 1) % 9);
        impl_->taskChoice->setCurrentIndex(taskIndex);
        if (impl_->enginePath->text() != previousEngine || state.prediction.overlay.empty() || !state.initialized)
            throw std::runtime_error("切换模型后状态未正确恢复");
        // 让隐藏行和多行模型信息完成布局，确保截图记录的是最终可见状态。
        QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        impl_->report("集成检查完成：" + state.prediction.details);
        writeImage(QDir(outputDirectory).filePath("result.png"), state.prediction.overlay);
        if (!state.prediction.mask.empty()) writeImage(QDir(outputDirectory).filePath("mask.png"), state.prediction.mask);
        if (!state.prediction.heatmap.empty()) writeImage(QDir(outputDirectory).filePath("heatmap.png"), state.prediction.heatmap);
        grab().save(QDir(outputDirectory).filePath("window.png"));
        QFile logFile(QDir(outputDirectory).filePath("ui.log"));
        if (!logFile.open(QIODevice::WriteOnly)) throw std::runtime_error("无法保存界面检查日志");
        logFile.write(impl_->log->toPlainText().toUtf8());
        return 0;
    } catch (const std::exception& error) {
        impl_->report(QString::fromUtf8(error.what()), true);
        grab().save(QDir(outputDirectory).filePath("failed-window.png"));
        QFile logFile(QDir(outputDirectory).filePath("ui.log"));
        if (logFile.open(QIODevice::WriteOnly)) logFile.write(impl_->log->toPlainText().toUtf8());
        return 1;
    }
}
