"""开发期验证C++抗锯齿缩放和ONNX图兼容变换；仅测试需要onnx/onnxruntime，不是产品依赖。"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort


def verify(executable: Path, models: Path, output: Path) -> None:
    """对三个已校准模型及四种长宽图像验证C++预处理，输出逐值误差报告。

    Args:
        executable: 已编译的anomaly_validation可执行程序。
        models: 包含三种ONNX和verification样本的目录。
        output: 兼容图、临时张量和验证报告的保存目录。
    """
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    for name in ("anomalydino", "dinomaly", "efficientad"):
        model = onnx.load(models / f"{name}.onnx")
        metadata = {item.key: item.value for item in model.metadata_props}
        height = int(metadata["anomalib.resize_height"])
        width = int(metadata["anomalib.resize_width"])
        prepared = output / f"{name}-prepared.onnx"
        subprocess.run([str(executable), "prepare", str(models / f"{name}.onnx"), str(prepared), str(height), str(width)], check=True)
        onnx.checker.check_model(prepared)
        resize = next(node for node in model.graph.node if node.name == "deployment_antialiased_resize")
        initializers = [item for item in model.graph.initializer if item.name in resize.input]
        graph = onnx.helper.make_graph([resize], "resize", [model.graph.input[0]],
                                      [onnx.helper.make_tensor_value_info(resize.output[0], onnx.TensorProto.FLOAT, [1, 3, height, width])], initializers)
        resize_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
        resize_model.ir_version = 8
        session = ort.InferenceSession(resize_model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        for index in range(4):
            image_path = models / "verification" / f"sample_{index}.png"
            image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
            blob = output / f"{name}-{index}.bin"
            subprocess.run([str(executable), "preprocess", str(image_path), str(width), str(height),
                            metadata["anomalib.resize_interpolation"], str(blob)], check=True)
            source = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
            expected = session.run(None, {"image": source})[0]
            actual = np.fromfile(blob, np.float32).reshape(expected.shape)
            maximum = float(np.max(np.abs(actual - expected)))
            mean = float(np.mean(np.abs(actual - expected)))
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            reports.append({"model": name, "sample": index, "maximum_abs_error": maximum, "mean_abs_error": mean})
        # 对非正方形真实缺陷图，原图内Resize与外移后的C++缩放/兼容图须输出一致。
        image = cv2.imread(str(models / "verification" / "sample_1.png"))
        source = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        original = ort.InferenceSession(str(models / f"{name}.onnx"), options, providers=["CPUExecutionProvider"])
        reference = original.run(None, {"image": source})
        del original
        transformed = ort.InferenceSession(str(prepared), options, providers=["CPUExecutionProvider"])
        resized = np.fromfile(output / f"{name}-1.bin", np.float32).reshape((1, 3, height, width))
        prediction = transformed.run(None, {transformed.get_inputs()[0].name: resized})
        score_error = float(np.max(np.abs(reference[0] - prediction[0])))
        map_error = float(np.max(np.abs(reference[1] - prediction[1])))
        np.testing.assert_allclose(prediction[0], reference[0], atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(prediction[1], reference[1], atol=1e-3, rtol=1e-3)
        reports.append({"model": name, "prepared_graph_score_error": score_error, "prepared_graph_map_error": map_error})
    (output / "preprocessing-report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


def main() -> None:
    """解析验证程序、模型与输出路径并执行开发期数值检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.executable.resolve(), args.models.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
