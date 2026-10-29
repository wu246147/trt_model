#include "mainwindow.h"
#include <QApplication>
#include <QTimer>
#include <iostream>

/// 启动统一窗口；--smoke-test 执行整图检查，--smoke-test-cut 在相同参数后增加切片尺寸及重叠像素。
int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("Raivas");
    QCoreApplication::setApplicationName("UnifiedTensorRT8616");
    const QStringList arguments = app.arguments();
    const bool cutCheck = arguments.contains("--smoke-test-cut");
    const bool check = arguments.contains("--smoke-test") || cutCheck;
    MainWindow window(nullptr, !check);
    window.show();
    if (check) {
        if (arguments.size() != (cutCheck ? 8 : 6) || arguments[1] != (cutCheck ? "--smoke-test-cut" : "--smoke-test")) {
            std::cerr << "用法：defect_alg_trt --smoke-test taskIndex(0..8) engine image outputDir\n"
                << "切片：defect_alg_trt --smoke-test-cut taskIndex engine image outputDir cutSize overlap\n";
            return 2;
        }
        bool valid = false;
        const int index = arguments[2].toInt(&valid);
        if (!valid || index < 0 || index > 8) return 2;
        int cutSize = 640, overlap = 64;
        if (cutCheck) {
            cutSize = arguments[6].toInt(&valid);
            if (!valid || cutSize < 32 || cutSize > 16384) return 2;
            overlap = arguments[7].toInt(&valid);
            if (!valid || overlap < 0 || overlap >= cutSize) return 2;
        }
        QTimer::singleShot(200, &window, [&app, &window, arguments, index, cutCheck, cutSize, overlap] {
            app.exit(window.smokeTest(index, arguments[3], arguments[4], arguments[5], cutCheck, cutSize, overlap));
        });
    }
    return app.exec();
}
