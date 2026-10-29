#include "imageview.h"
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QWheelEvent>
#include <cmath>

/// 建立由控件管理的场景，避免窗口关闭后遗留图像资源。
ImageView::ImageView(QWidget* parent) : QGraphicsView(parent) {
    setScene(new QGraphicsScene(this));
    imageItem_ = scene()->addPixmap(QPixmap());
    setBackgroundBrush(QColor(40, 44, 50));
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setRenderHint(QPainter::SmoothPixmapTransform);
    setMinimumSize(260, 240);
}

/// 将输入图像复制到场景，在尺寸变化或适应模式下重新计算缩放。
void ImageView::setImage(const QImage& image) {
    const bool changed = imageItem_->pixmap().size() != image.size();
    imageItem_->setPixmap(QPixmap::fromImage(image));
    scene()->setSceneRect(imageItem_->boundingRect());
    if (changed || fitMode_) fitImage();
}

/// 计算保持长宽比的整图缩放，空图像无需处理。
void ImageView::fitImage() {
    if (!imageItem_->pixmap().isNull()) fitInView(imageItem_, Qt::KeepAspectRatio);
    fitMode_ = true;
}

/// 将滚轮角度映射为连续缩放系数，限制范围以防图像失去可见性。
void ImageView::wheelEvent(QWheelEvent* event) {
    if (imageItem_->pixmap().isNull()) return;
    const double factor = std::pow(1.0015, event->angleDelta().y());
    const double zoom = transform().m11() * factor;
    if (zoom >= 0.01 && zoom <= 64.0) {
        scale(factor, factor);
        fitMode_ = false;
    }
    event->accept();
}

/// 双击时恢复完整图像范围。
void ImageView::mouseDoubleClickEvent(QMouseEvent* event) {
    fitImage();
    event->accept();
}

/// 先执行基类尺寸更新，然后按整图适应模式刷新显示。
void ImageView::resizeEvent(QResizeEvent* event) {
    QGraphicsView::resizeEvent(event);
    if (fitMode_) fitImage();
}
