"""用原训练环境保存的golden结果验证三个TensorRT异常检测引擎，不需要Python推理库。"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def verify(executable: Path, engines: Path, fixtures: Path, report: Path) -> None:
    """逐模型、逐样本运行原生验证并保存误差、一致率及推理耗时。

    Args:
        executable: anomaly_validation可执行程序。
        engines: 保存模型名-fp32.engine的目录。
        fixtures: 包含sample_0..3.png及对应模型golden yml的目录。
        report: JSON验证报告路径；每项完成立即更新，失败保留已有结果。
    """
    rows = []
    for name in ("efficientad", "anomalydino", "dinomaly"):
        for index in range(4):
            command = [str(executable), "infer", name, str(engines / f"{name}-fp32.engine"),
                       str(fixtures / f"sample_{index}.png"), str(fixtures / f"{name}_{index}.yml")]
            process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
            lines = [line.strip().lstrip("\ufeff") for line in process.stdout.splitlines() if line.strip().lstrip("\ufeff").startswith("{")]
            row = {"model": name, "sample": index, "passed": process.returncode == 0 and bool(lines)}
            if lines:
                row.update(json.loads(lines[-1]))
            if process.returncode or not lines:
                row["error"] = process.stderr.strip() or process.stdout.strip()
            rows.append(row)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"precision": "FP32, TF32 disabled", "samples": rows}, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
            if process.returncode or not lines:
                raise RuntimeError(f"{name}样本{index}验证失败: {row.get('error', '')}")


def main() -> None:
    """解析原生验证程序、引擎、样本及报告路径并执行全量检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("engines", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    verify(args.executable.resolve(), args.engines.resolve(), args.fixtures.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
