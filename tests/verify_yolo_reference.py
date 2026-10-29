"""开发验证：以同一 letterbox 输入对照 ONNX Runtime CPU 和 TensorRT FP32；此依赖不进入发布包。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort


def arguments():
    """读取验证模型、图片、原生工具及误差边界；默认使用五个本机已有真实权重模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("validation"))
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["yolov5s", "yolo11n", "yolo11s-seg", "yolo26s", "yolo26s-seg"])
    parser.add_argument("--dll-directory", type=Path, action="append", default=[])
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def make_input(image, shape):
    """按照 C++ 接口的 114 填边规则生成 FP32 RGB NCHW，固定批次剩余槽位填零。"""
    if len(shape) != 4 or any(not isinstance(value, int) or value <= 0 for value in shape) or shape[1] != 3:
        raise ValueError(f"参考验证需要静态 NCHW 三通道 ONNX，实际为 {shape}")
    batch, _, height, width = shape
    gain = min(width / image.shape[1], height / image.shape[0])
    content_width = int(math.floor(image.shape[1] * gain + 0.5))
    content_height = int(math.floor(image.shape[0] * gain + 0.5))
    left = int(math.floor((width - content_width) / 2 - 0.1 + 0.5))
    top = int(math.floor((height - content_height) / 2 - 0.1 + 0.5))
    resized = cv2.resize(image, (content_width, content_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, height - content_height - top, left, width - content_width - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    data = np.zeros((batch, 3, height, width), dtype=np.float32)
    data[0] = np.transpose(padded[:, :, ::-1].astype(np.float32) / np.float32(255), (2, 0, 1))
    return data


def error_metrics(reference, actual, atol, rtol):
    """记录全张量绝对误差分布及 allclose 结果；相对误差项使用参考值幅度避免低分除零。"""
    if reference.shape != actual.shape:
        return {"passed": False, "reference_shape": list(reference.shape), "actual_shape": list(actual.shape)}
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        return {"passed": False, "error": "输出包含 NaN 或无穷值"}
    difference = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    limit = atol + rtol * np.abs(reference.astype(np.float64))
    return {"passed": bool(np.all(difference <= limit)), "elements": int(reference.size),
        "max_absolute": float(difference.max(initial=0)), "mean_absolute": float(difference.mean()),
        "p99_absolute": float(np.quantile(difference, 0.99)), "outside_tolerance": int(np.count_nonzero(difference > limit))}


def overlap(first, second):
    """计算两个 xyxy 框的 IoU，空框或无交集时返回零。"""
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    union = max(0, first[2] - first[0]) * max(0, first[3] - first[1]) + \
        max(0, second[2] - second[0]) * max(0, second[3] - second[1]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def compare_end_to_end(reference, actual, atol, rtol):
    """按类别和 IoU 匹配置信度严格大于 0.5 的实例，避开 TopK 低分近似并列导致的行序变化。"""
    batches = []
    passed = True
    for batch in range(reference.shape[0]):
        expected = reference[batch][reference[batch, :, 4] > 0.5]
        found = actual[batch][actual[batch, :, 4] > 0.5]
        current = {"batch": batch, "reference_count": len(expected), "engine_count": len(found), "matches": []}
        available = set(range(len(found)))
        matched = len(expected) == len(found)
        for row in expected:
            candidates = [index for index in available if found[index, 5] == row[5]]
            if not candidates:
                matched = False
                current["matches"].append({"class_id": int(row[5]), "error": "无同类匹配目标"})
                continue
            index = max(candidates, key=lambda item: overlap(row[:4], found[item, :4]))
            available.remove(index)
            target = found[index]
            score_error = float(abs(row[4] - target[4]))
            coordinate_error = float(np.max(np.abs(row[:4] - target[:4])))
            iou = overlap(row[:4], target[:4])
            coefficients = error_metrics(row[6:], target[6:], atol, rtol) if row.size > 6 else {"passed": True}
            valid = iou >= 0.995 and coordinate_error <= 0.25 and score_error <= 5e-4 and coefficients["passed"]
            matched = matched and valid
            current["matches"].append({"class_id": int(row[5]), "iou": iou, "max_coordinate_error_pixels": coordinate_error,
                "score_error": score_error, "coefficients": coefficients, "passed": bool(valid)})
        current["passed"] = bool(matched)
        batches.append(current)
        passed = passed and matched
    return {"passed": bool(passed), "confidence": 0.5, "max_coordinate_error_pixels": 0.25,
        "minimum_iou": 0.995, "max_score_error": 5e-4, "batches": batches,
        "reason": "TopK 背景低分的排序不影响业务目标；按固定业务阈值逐目标比较坐标、分数、类别和掩码系数"}


def verify_model(options, name, image, environment):
    """运行同一模型的 CPU 参考和原生 engine，按输出名称对齐张量并汇总验证证据。"""
    model = options.directory / (name + ".onnx")
    engine = options.directory / (name + ".engine")
    if not model.is_file() or not engine.is_file():
        raise FileNotFoundError(f"需要已构建的同名 ONNX 和 engine：{model} / {engine}")
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    settings.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=settings, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    data = make_input(image, input_info.shape)
    input_file = options.directory / (name + ".reference_input.bin")
    data.tofile(str(input_file))
    started = time.perf_counter()
    expected = dict(zip([value.name for value in session.get_outputs()], session.run(None, {input_info.name: data})))
    cpu_seconds = time.perf_counter() - started
    prefix = options.directory / (name + ".reference_trt")
    command = [str(options.tool.resolve()), "infer", str(engine.resolve()), str(input_file.resolve()), str(prefix.resolve())]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, encoding="utf-8", errors="replace", timeout=180)
    if process.returncode:
        raise RuntimeError(f"原生推理失败 {process.returncode}: {process.stdout}\n{process.stderr}")
    bindings = []
    for line in process.stdout.splitlines():
        match = re.fullmatch(r"output (\S+) ((?:[0-9]+\s*)+)", line)
        if match:
            bindings.append((match.group(1), tuple(int(value) for value in match.group(2).split())))
    if len(bindings) != len(expected) or {value[0] for value in bindings} != set(expected):
        raise ValueError(f"原生输出与 ONNX 输出名称不一致：{bindings} / {list(expected)}")
    metadata = session.get_modelmeta().custom_metadata_map
    end_to_end = metadata.get("end2end", "false").lower() in ("true", "1")
    outputs = []
    for index, (output_name, shape) in enumerate(bindings):
        actual = np.fromfile(str(prefix) + f".{index}.bin", dtype=np.float32)
        if actual.size != math.prod(shape):
            raise ValueError(f"原生输出二进制数量与尺寸不一致：{output_name} / {actual.size} / {shape}")
        actual = actual.reshape(shape)
        reference = expected[output_name]
        statistics = error_metrics(reference, actual, options.atol, options.rtol)
        record = {"name": output_name, "shape": list(shape), "raw_tensor": statistics}
        if end_to_end and reference.ndim == 3 and reference.shape[-1] >= 6:
            semantic = compare_end_to_end(reference, actual, options.atol, options.rtol)
            record["matched_instances"] = semantic
            record["passed"] = semantic["passed"]
        else:
            record["passed"] = statistics["passed"]
        outputs.append(record)
    return {"model": name, "passed": all(output["passed"] for output in outputs), "input_shape": list(data.shape),
        "class_count": metadata.get("class_count"), "end2end": end_to_end,
        "onnx_runtime_cpu_seconds": cpu_seconds, "outputs": outputs}


def main():
    """验证全部已指定真实模型并保存 JSON；任一错误以非零退出码返回，不跳过缺失模型。"""
    options = arguments()
    if options.atol < 0 or options.rtol < 0:
        raise ValueError("误差阈值不能为负数")
    image = cv2.imdecode(np.fromfile(str(options.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码验证图片：{options.image}")
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(str(path.resolve()) for path in options.dll_directory) + os.pathsep + environment.get("PATH", "")
    report = {"onnxruntime": ort.__version__, "provider": "CPUExecutionProvider", "image": str(options.image.resolve()),
        "image_sha256": hashlib.sha256(options.image.read_bytes()).hexdigest(), "precision": "TensorRT FP32, TF32 disabled",
        "absolute_tolerance": options.atol, "relative_tolerance": options.rtol,
        "tolerance_reason": "原始坐标允许相对 FP32 误差，接近零的概率以绝对误差约束；end2end 另按 0.5 固定置信度匹配实例",
        "models": []}
    for name in options.models:
        try:
            result = verify_model(options, name, image, environment)
        except Exception as error:
            result = {"model": name, "passed": False, "error": str(error)}
        report["models"].append(result)
        print(f"{name}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    report["passed"] = all(model["passed"] for model in report["models"])
    destination = options.report or options.directory / "yolo_reference_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination.resolve())
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
