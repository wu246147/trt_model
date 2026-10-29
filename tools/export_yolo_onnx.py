"""仅用于开发机：使用已有官方源码和本地 PT 导出静态 ONNX，部署程序不依赖 Python。"""

import argparse
import json
import os
from pathlib import Path
import sys


def arguments():
    """读取版本、官方源码目录和已有权重；不自动下载权重或安装任何依赖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=(5, 11, 26), required=True)
    parser.add_argument("--repo", type=Path, required=True, help="本机 YOLOv5 或 Ultralytics 官方源码根目录")
    parser.add_argument("--weights", type=Path, required=True, help="已经存在的本地 .pt 权重")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, choices=(12, 13), default=13)
    parser.add_argument("--end2end", choices=("auto", "on", "off"), default="auto",
        help="YOLO11/26 输出分支；auto 遵循官方 YAML，on 要求权重包含 one-to-one 分支")
    return parser.parse_args()


def load_model(options):
    """从指定官方源码加载本地 PT 并设置标准导出模式；返回模型、类别字典及任务。"""
    if not options.repo.is_dir() or not options.weights.is_file():
        raise FileNotFoundError("官方源码目录和本地权重必须已经存在；本工具不联网下载")
    sys.path.insert(0, str(options.repo.resolve()))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    # 部分官方加载流程会保存设置；将开发工具缓存限制到输出目录，避免修改用户全局环境。
    configuration = options.output.parent.resolve() / ".ultralytics"
    configuration.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(configuration)
    import torch

    if options.version == 5:
        if not (options.repo / "models" / "yolo.py").is_file():
            raise ValueError("YOLO5 必须指定 YOLOv5 官方源码根目录")
        from models.experimental import attempt_load
        from models.yolo import Detect

        model = attempt_load(str(options.weights.resolve()), device=torch.device("cpu"), inplace=True, fuse=True)
        for module in model.modules():
            if isinstance(module, Detect):
                module.inplace = True
                module.dynamic = False
                module.export = True
    else:
        if not (options.repo / "ultralytics" / "__init__.py").is_file():
            raise ValueError("YOLO11/26 必须指定 Ultralytics 官方源码根目录")
        from ultralytics import YOLO
        from ultralytics.nn.modules import Detect

        wrapper = YOLO(str(options.weights.resolve()))
        model = wrapper.model
        head = model.model[-1]
        # 部分旧 checkpoint 没有新版属性需要的 _end2end；融合前恢复 YAML 中的正式分支选择。
        desired_end_to_end = bool(model.yaml.get("end2end", options.version == 26))
        if options.end2end != "auto":
            desired_end_to_end = options.end2end == "on"
        if desired_end_to_end and getattr(head, "one2one_cv2", None) is None:
            raise ValueError("所选权重没有 one-to-one 分支，无法导出 end2end 模型")
        if hasattr(head, "end2end"):
            head.end2end = desired_end_to_end
        model.fuse()
        for module in model.modules():
            if isinstance(module, Detect):
                module.export = True
                module.dynamic = False
                module.format = "onnx"
    model = model.float().cpu().eval()
    head = model.model[-1]
    task = "segment" if hasattr(head, "proto") and hasattr(head, "nm") else "detect"
    if task == "detect" and getattr(head, "kpt_shape", None) is not None:
        raise ValueError("当前接口只接受目标检测和实例分割权重")
    names = model.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    return model, names, task, bool(getattr(head, "end2end", False))


def main():
    """导出静态 FP32 输入输出 ONNX，写入自动读取所需元数据并校验后原子替换目标文件。"""
    options = arguments()
    if options.batch <= 0 or options.height <= 0 or options.width <= 0:
        raise ValueError("批次和输入宽高必须为正数")
    if options.height % 32 or options.width % 32:
        raise ValueError("标准 YOLO 输入宽高必须是 32 的整数倍")
    import onnx
    import torch

    torch.set_num_threads(4)
    model, names, task, end_to_end = load_model(options)
    sample = torch.zeros(options.batch, 3, options.height, options.width, dtype=torch.float32)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = options.output.with_name(options.output.stem + ".exporting.onnx")
    outputs = ["output0", "output1"] if task == "segment" else ["output0"]
    with torch.inference_mode():
        model(sample)
        model(sample)
        torch.onnx.export(
            model, sample, str(temporary), export_params=True,
            opset_version=options.opset, do_constant_folding=True,
            input_names=["images"], output_names=outputs, dynamic_axes=None,
        )
    exported = onnx.load(str(temporary))
    metadata = {
        "names": json.dumps(names, ensure_ascii=False),
        "class_count": str(len(names)),
        "batch": str(options.batch),
        "imgsz": json.dumps([options.height, options.width]),
        "task": task,
        "model_family": f"yolo{options.version}",
        "end2end": str(end_to_end).lower(),
        "args": json.dumps({"end2end": end_to_end, "dynamic": False, "opset": options.opset}),
    }
    onnx.helper.set_model_props(exported, metadata)
    exported = onnx.shape_inference.infer_shapes(exported)
    onnx.checker.check_model(exported)
    onnx.save(exported, str(temporary))
    temporary.replace(options.output)
    print(json.dumps({"onnx": str(options.output.resolve()), "task": task, "classes": len(names),
        "input": [options.batch, 3, options.height, options.width], "end2end": end_to_end}, ensure_ascii=False))


if __name__ == "__main__":
    main()
