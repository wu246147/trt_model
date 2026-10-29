#pragma once
#include <QGraphicsView>
#include <QImage>

class QGraphicsPixmapItem;

/// 图像显示控件，支持滚轮缩放、拖动平移和双击适应窗口。
class ImageView final : public QGraphicsView {
public:
    /// 创建独立图像场景；parent 为拥有此控件的父窗口。
    explicit ImageView(QWidget* parent = nullptr);
    /// 显示图像副本；图像大小不变时保留用户的缩放位置。
    void setImage(const QImage& image);
    /// 使当前图像完整显示，并启用窗口尺寸变化时自动适应。
    void fitImage();
protected:
    /// 根据滚轮步长缩放，以鼠标位置为中心并限制缩放范围。
    void wheelEvent(QWheelEvent* event) override;
    /// 双击恢复整图显示，消费此次鼠标事件。
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    /// 窗口大小改变时按当前适应模式更新图像比例。
    void resizeEvent(QResizeEvent* event) override;
private:
    QGraphicsPixmapItem* imageItem_ = nullptr;
    bool fitMode_ = true;
};
