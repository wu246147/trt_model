"""开发验证：用真实 YOLO11 分割 ONNX 检查奇数尺寸原图的原型裁剪是否符合官方 scale_masks。"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

from verify_yolo_reference import make_input


def crop_regions(original, target, prototype):
    """返回旧的输入 padding 下采样区域与修复后的原型分辨率区域，用于隔离一行/列取整误差。"""
    width, height = original
    target_width, target_height = target
    proto_width, proto_height = prototype
    rounded = lambda value: int(math.floor(value + 0.5))
    gain = min(target_width / width, target_height / height)
    content_width, content_height = rounded(width * gain), rounded(height * gain)
    left = rounded((target_width - content_width) / 2 - 0.1)
    top = rounded((target_height - content_height) / 2 - 0.1)
    old = (rounded(left * proto_width / target_width - 0.1), rounded(top * proto_height / target_height - 0.1),
        rounded((left + content_width) * proto_width / target_width + 0.1),
        rounded((top + content_height) * proto_height / target_height + 0.1))
    proto_gain = min(proto_width / width, proto_height / height)
    new_width, new_height = round(width * proto_gain), round(height * proto_gain)
    new_left = round((proto_width - new_width) / 2 - 0.1)
    new_top = round((proto_height - new_height) / 2 - 0.1)
    return old, (new_left, new_top, new_left + new_width, new_top + new_height)


def resize_mask(logits, region, size):
    """按给定有效原型区域进行双线性缩放并以零 logit 二值化，对应 C++ 默认概率阈值 0.5。"""
    left, top, right, bottom = region
    return cv2.resize(logits[top:bottom, left:right], size, interpolation=cv2.INTER_LINEAR) > 0


def main():
    """运行多个奇数原图尺寸并保存旧算法、修复算法与官方函数的实际像素差异。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("validation/yolo11s-seg.onnx"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("validation/yolo_mask_scaling_report.json"))
    options = parser.parse_args()
    sys.path.insert(0, str(options.repo.resolve()))
    os.environ["YOLO_CONFIG_DIR"] = str(options.report.parent.resolve() / ".ultralytics")
    from ultralytics.utils.ops import process_mask_native

    torch.set_num_threads(4)
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    session = ort.InferenceSession(str(options.model), sess_options=settings, providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    image = cv2.imdecode(np.fromfile(str(options.image), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取验证图片")
    report = {"model": str(options.model), "cases": []}
    for width, height in ((101, 103), (101, 125), (101, 191), (347, 513), (513, 347)):
        source = cv2.resize(image, (width, height))
        outputs = session.run(None, {session.get_inputs()[0].name: make_input(source, shape)})
        prediction, prototype = outputs[0][0], outputs[1][0]
        mask_channels = prototype.shape[0]
        classes = prediction.shape[0] - 4 - mask_channels
        anchor = int(np.argmax(prediction[4:4 + classes].max(axis=0)))
        coefficients = prediction[4 + classes:, anchor]
        logits = (coefficients[None] @ prototype.reshape(mask_channels, -1)).reshape(prototype.shape[1:])
        official = process_mask_native(torch.from_numpy(prototype), torch.from_numpy(coefficients[None]),
            torch.tensor([[0, 0, width, height]], dtype=torch.float32), (height, width))[0].numpy() > 0
        old_region, new_region = crop_regions((width, height), (shape[3], shape[2]), (prototype.shape[2], prototype.shape[1]))
        old = resize_mask(logits, old_region, (width, height))
        new = resize_mask(logits, new_region, (width, height))
        old_errors = int(np.count_nonzero(old != official))
        new_errors = int(np.count_nonzero(new != official))
        union = int(np.count_nonzero(new | official))
        intersection = int(np.count_nonzero(new & official))
        iou = intersection / union if union else 1.0
        report["cases"].append({"original": [width, height], "old_region": old_region, "corrected_region": new_region,
            "old_different_pixels": old_errors, "corrected_different_pixels": new_errors,
            "corrected_iou": iou, "passed": iou >= 0.999})
    report["passed"] = all(case["passed"] for case in report["cases"])
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
