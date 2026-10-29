"""生成六种可确定预期结果的小型 ONNX，用于验证解码、批次、坐标和分割边界。"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def model(version, segmentation):
    """构造 batch=2、64×96 输入的固定预测，包含同类重叠、异类重叠、低分和无效候选。"""
    end_to_end = version == 26
    masks = 2 if segmentation else 0
    channels = (6 if end_to_end else (7 if version == 5 else 6)) + masks
    rows = np.zeros((2, 8, channels), dtype=np.float32)
    for batch in range(2):
        for anchor in range(8):
            if end_to_end:
                rows[batch, anchor, :4] = (32, 20, 64, 44)
            else:
                rows[batch, anchor, :4] = (48, 32, 32, 24)
            if masks:
                rows[batch, anchor, -2:] = (2, 0)
        if end_to_end:
            rows[batch, :, 4] = (0.9, 0.85, 0.8, 0.1, 0.9, 0.49, 0.9, 0)
            rows[batch, :, 5] = (0, 0, 1, 0, 0, 0, 2, 0)
            rows[batch, 4, 2] = 20  # x2 小于 x1，应被丢弃。
        else:
            start = 5 if version == 5 else 4
            rows[batch, :, start] = (0.9, 0.85, 0.05, 0.1, 0.9, 0.49, 1.4, 0)
            rows[batch, 2, start + 1] = 0.8
            rows[batch, 4, 2] = -2  # 负宽度不能进入 NMS 或坐标变换。
            if version == 5:
                rows[batch, :, 4] = (0.8, 0.8, 0.8, 0.1, 0.8, 0.8, 0.8, 0)
    if version == 11:
        rows = rows.transpose(0, 2, 1).copy()
    # 对输入均值保留极小依赖，确保不是完全忽略输入的常量网络；只扰动分数，不扰动类别 ID。
    score_mask = np.zeros_like(rows)
    if version == 11:
        score_mask[:, 4, :] = 1
    else:
        score_mask[:, :, 4 if end_to_end or version == 5 else 4] = 1
    initializers = [
        numpy_helper.from_array(rows, "base"),
        numpy_helper.from_array(score_mask, "score_mask"),
        numpy_helper.from_array(np.array([1e-8], dtype=np.float32), "epsilon"),
        numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "unsqueeze_axes"),
    ]
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Unsqueeze", ["mean", "unsqueeze_axes"], ["expanded"]),
        helper.make_node("Mul", ["expanded", "epsilon"], ["tiny"]),
        helper.make_node("Mul", ["tiny", "score_mask"], ["shift"]),
        helper.make_node("Add", ["base", "shift"], ["output0"]),
    ]
    outputs = [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(rows.shape))]
    if segmentation:
        prototype = np.ones((2, 2, 16, 24), dtype=np.float32)
        prototype[:, 1] = -1
        initializers.append(numpy_helper.from_array(prototype, "prototype"))
        nodes.append(helper.make_node("Identity", ["prototype"], ["output1"]))
        outputs.append(helper.make_tensor_value_info("output1", TensorProto.FLOAT, list(prototype.shape)))
    graph = helper.make_graph(nodes, f"yolo{version}_{'seg' if segmentation else 'det'}_synthetic",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [2, 3, 64, 96])], outputs, initializers)
    result = helper.make_model(graph, producer_name="trt_model_validation", opset_imports=[helper.make_opsetid("", 13)])
    result.ir_version = 9
    helper.set_model_props(result, {"names": json.dumps({0: "normal", 1: "defect"}), "class_count": "2",
        "batch": "2", "imgsz": "[64,96]", "end2end": str(end_to_end).lower(),
        "args": json.dumps({"end2end": end_to_end}), "task": "segment" if segmentation else "detect"})
    onnx.checker.check_model(result)
    return result


def main():
    """将六种测试模型写入指定目录；生成过程只需要开发机的 numpy/onnx，不参与部署。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    options.directory.mkdir(parents=True, exist_ok=True)
    for version in (5, 11, 26):
        for segmentation in (False, True):
            filename = options.directory / f"yolo{version}_{'seg' if segmentation else 'det'}.onnx"
            onnx.save(model(version, segmentation), str(filename))
            print(filename)


if __name__ == "__main__":
    main()
