"""开发期对照 Dinomaly 的普通 FP16 与 FP32 精度保护，产品不依赖本脚本或 Python。"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_samples(executable: Path, engine: Path, fixtures: Path) -> list[dict]:
    """用四个参考样本执行原生校验，保留成功与失败结果供修正前后对照。

    Args:
        executable: anomaly_validation 可执行文件。
        engine: 待验证的 TensorRT 引擎。
        fixtures: 原训练环境保存的图片和 Dinomaly 参考输出目录。
    Returns:
        每张图的分数、分类、异常图误差、掩码一致率和原生退出码。
    """
    rows = []
    for index in range(4):
        process = subprocess.run(
            [str(executable), "infer", "dinomaly", str(engine),
             str(fixtures / f"sample_{index}.png"), str(fixtures / f"dinomaly_{index}.yml")],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        lines = [line.lstrip("\ufeff") for line in process.stdout.splitlines()
                 if line.lstrip("\ufeff").startswith("{")]
        if not lines:
            raise RuntimeError(f"样本 {index} 没有返回有效结果，退出码 {process.returncode}: {process.stderr or process.stdout}")
        row = {"sample": index, "passed": process.returncode == 0, "exit_code": process.returncode}
        row.update(json.loads(lines[-1]))
        if process.returncode:
            row["error"] = process.stderr.strip()
        rows.append(row)
    return rows


def inspect_ranges(model_path: Path, image_path: Path) -> list[dict]:
    """提取线性注意力分母及未归一化乘积的 FP32 范围，检测超过 FP16 表示上限的元素。

    Args:
        model_path: 含图内预处理的原始 Dinomaly ONNX，只读不修改。
        image_path: 原图，按原始协议转换为 RGB、NCHW、0..1。
    Returns:
        中间张量的形状、范围及超过 65504 的元素数量；仅开发分析需要 ONNX/ORT。
    """
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(model_path))
    producers = {output: node for node in model.graph.node for output in node.output}
    names = []
    for node in model.graph.node:
        if node.op_type != "Mul" or len(node.input) != 2:
            continue
        sources = [producers.get(name) for name in node.input]
        reciprocal = next((value for value in sources if value is not None and value.op_type == "Reciprocal"), None)
        product = next((value for value in sources if value is not None and value.op_type == "MatMul"), None)
        if reciprocal is not None and product is not None:
            names.extend([reciprocal.input[0], product.output[0]])
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("模型中未找到带倒数归一化的线性注意力，不能据此解释 FP16 风险")
    for name in names:
        model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    session = ort.InferenceSession(model.SerializeToString(), settings, providers=["CPUExecutionProvider"])
    image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    data = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    outputs = session.run(names, {session.get_inputs()[0].name: data})
    return [{"tensor": name, "shape": list(value.shape), "minimum": float(value.min()),
             "maximum": float(value.max()), "over_fp16_max": int(np.count_nonzero(np.abs(value) > 65504)),
             "elements": int(value.size)} for name, value in zip(names, outputs)]


def main() -> int:
    """保存 FP32、普通 FP16、精度保护版的同样本结果，修正后任一失败即返回非零。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ranges-onnx", type=Path)
    options = parser.parse_args()
    report = {"model": "Dinomaly", "policy": "FP32 protection; original FP16 request is explicitly overridden",
              "engines": {key: str(getattr(options, key).resolve()) for key in ("fp32", "before", "after")}}
    for key in ("fp32", "before", "after"):
        report[key] = run_samples(options.executable.resolve(), getattr(options, key).resolve(), options.fixtures.resolve())
    if options.ranges_onnx:
        report["fp32_attention_ranges"] = inspect_ranges(options.ranges_onnx, options.fixtures / "sample_1.png")
    report["passed"] = all(row["passed"] for row in report["fp32"] + report["after"])
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("fp32", "before", "after"):
        print(key, [(row["sample"], row["score"], row["passed"]) for row in report[key]])
    print(options.report.resolve())
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
