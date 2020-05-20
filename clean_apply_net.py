import glob
import os
from typing import Any, Dict

import torch

from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.data.detection_utils import convert_PIL_to_numpy, read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.boxes import BoxMode
from detectron2.structures.instances import Instances


def execute2(args: dict):
    print(f"Loading config from {args['cfg']}")
    cfg = setup_config(args['cfg'], args['model'], args)

    print(f"Loading model from {args['model']}")
    predictor = DefaultPredictor(cfg)

    print(f"Loading data from {args['input']}")
    pils = [args['input']]
    if len(pils) == 0:
        print(f"No input images for {args['input']}")
        return
    context = create_context(args)
    for pil in pils:
        # predictor expects BGR image.
        img = convert_PIL_to_numpy(pil, format="BGR")
        with torch.no_grad():
            outputs = predictor(img)["instances"]
            execute_on_outputs(
                context, {"file_name": "pil", "image": img}, outputs)
    return context["results"]


def execute(args: dict):
    print(f"Loading config from {args['cfg']}")
    cfg = setup_config(args['cfg'], args['model'], args)

    print(f"Loading model from {args['model']}")
    predictor = DefaultPredictor(cfg)

    print(f"Loading data from {args['input']}")
    file_list = _get_input_file_list(args['input'])
    if len(file_list) == 0:
        print(f"No input images for {args['input']}")
        return
    context = create_context(args)
    for file_name in file_list:
        # predictor expects BGR image.
        img = read_image(file_name, format="BGR")
        with torch.no_grad():
            outputs = predictor(img)["instances"]
            execute_on_outputs(
                context, {"file_name": file_name, "image": img}, outputs)
    return context["results"]


def setup_config(config_fpath: str, model_fpath: str, args: dict):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.merge_from_list(args['opts'])
    cfg.MODEL.WEIGHTS = model_fpath
    cfg.freeze()
    return cfg


def _get_input_file_list(input_spec: str):
    if os.path.isdir(input_spec):
        file_list = [
            os.path.join(input_spec, fname)
            for fname in os.listdir(input_spec)
            if os.path.isfile(os.path.join(input_spec, fname))
        ]
    elif os.path.isfile(input_spec):
        file_list = [input_spec]
    else:
        file_list = glob.glob(input_spec)
    return file_list


def execute_on_outputs(context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances):
    image_fpath = entry["file_name"]
    print(f"Processing {image_fpath}")
    result = {"file_name": image_fpath}
    if outputs.has("scores"):
        result["scores"] = outputs.get("scores").cpu()
    if outputs.has("pred_boxes"):
        result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
        if outputs.has("pred_densepose"):
            boxes_XYWH = BoxMode.convert(
                result["pred_boxes_XYXY"], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
            )
            result["pred_densepose"] = outputs.get(
                "pred_densepose").to_result(boxes_XYWH)
    context["results"].append(result)


def create_context(args: dict):
    context = {"results": []}
    return context
