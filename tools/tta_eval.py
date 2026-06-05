"""
TTA evaluation (multi-scale + horizontal flip + merge + NMS).

Default policy (from thesis plan):
  - Scales: [800, 1000, 1200] (short side)
  - For each scale: original + hflip  => 6 forwards per image
  - Merge all detections back to original image coordinates
  - Class-wise NMS (IoU=0.5) and keep top max_dets (default 500)
  - Run COCO bbox evaluation

Example:
  python tools/tta_eval.py ^
    --resume outputs/final_stretch/checkpoint_best_ema.pth ^
    --config_file config/DINO/DINO_5scale_visdrone_final.py ^
    --val_ann_file path/to/instances_val2017.json ^
    --val_img_folder path/to/images ^
    --amp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TVF
from torchvision.ops import nms

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.transforms import resize as _resize_pil_target
from datasets.transforms import hflip as _hflip_pil_target
from util.misc import torch_load_trusted


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("tta_eval")
    p.add_argument("--resume", required=True, help="Checkpoint .pth to evaluate (regular/EMA).")
    p.add_argument(
        "--config_file",
        "-c",
        default=str(_repo_root() / "config" / "DINO" / "DINO_5scale_visdrone_final.py"),
        help="Config file used for evaluation (must match training).",
    )
    # Dataset inputs (same styles as main.py)
    p.add_argument("--val_img_folder", default=None, help="Val images folder (pair with --val_ann_file).")
    p.add_argument("--val_ann_file", default=None, help="Val COCO JSON file (pair with --val_img_folder).")
    p.add_argument("--dataset_json_root", default="", help="Alternative dataset root (contains annotations/...).")
    p.add_argument("--dataset_images", default="", help="Alternative images folder (pair with dataset_json_root).")
    p.add_argument("--coco_path", default="", help="Standard COCO root (train2017/val2017/annotations).")

    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--amp", action="store_true", help="Enable AMP during inference.")
    p.add_argument("--num_workers", type=int, default=4)

    # TTA params
    p.add_argument("--scales", default="800,1000,1200", help="Comma-separated short-side scales.")
    p.add_argument("--max_size", type=int, default=2000, help="Max long side after resize.")
    p.add_argument("--nms_iou", type=float, default=0.5, help="NMS IoU threshold for merging TTA results.")
    p.add_argument("--max_dets", type=int, default=500, help="Max detections per image after NMS.")

    p.add_argument(
        "--extra_options",
        nargs="*",
        default=[],
        help="Extra --options overrides (key=value) merged into config before building model/dataset.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_scales(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Empty --scales")
    return out


def _make_input_tensor(img_pil) -> torch.Tensor:
    # Same normalize as datasets/coco.py
    t = TVF.to_tensor(img_pil)
    t = TVF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return t


def _unflip_boxes_xyxy(boxes: torch.Tensor, width: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    x1 = boxes[:, 0]
    x2 = boxes[:, 2]
    boxes = boxes.clone()
    boxes[:, 0] = float(width) - x2
    boxes[:, 2] = float(width) - x1
    return boxes


def _rescale_boxes_xyxy(boxes: torch.Tensor, from_wh: Tuple[int, int], to_wh: Tuple[int, int]) -> torch.Tensor:
    """Scale xyxy boxes from (W_from,H_from) to (W_to,H_to)."""
    if boxes.numel() == 0:
        return boxes
    Wf, Hf = float(from_wh[0]), float(from_wh[1])
    Wt, Ht = float(to_wh[0]), float(to_wh[1])
    sx = Wt / max(Wf, 1.0)
    sy = Ht / max(Hf, 1.0)
    out = boxes.clone()
    out[:, 0] *= sx
    out[:, 2] *= sx
    out[:, 1] *= sy
    out[:, 3] *= sy
    return out


@torch.no_grad()
def _tta_predict_one(
    model,
    postprocessors,
    img_pil,
    device: torch.device,
    *,
    scales: List[int],
    max_size: int,
    amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns merged raw detections in ORIGINAL image xyxy coords."""
    W0, H0 = img_pil.size

    all_boxes: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for s in scales:
        # resize (no target needed for inference)
        img_r, _ = _resize_pil_target(img_pil, None, s, max_size)
        Wr, Hr = img_r.size

        for do_flip in (False, True):
            img_a = img_r
            if do_flip:
                img_a, _ = _hflip_pil_target(img_a, None)

            ta = _make_input_tensor(img_a).to(device)
            samples = utils.nested_tensor_from_tensor_list([ta])
            target_sizes = torch.as_tensor([[img_a.size[1], img_a.size[0]]], device=device)  # (H,W)

            with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
                outputs = model(samples)
                results = postprocessors["bbox"](outputs, target_sizes)

            pred = results[0]
            boxes = pred["boxes"]  # xyxy in augmented image abs coords
            scores = pred["scores"]
            labels = pred["labels"]

            if do_flip:
                boxes = _unflip_boxes_xyxy(boxes, width=Wr)

            # map resized -> original
            boxes = _rescale_boxes_xyxy(boxes, from_wh=(Wr, Hr), to_wh=(W0, H0))

            all_boxes.append(boxes.detach().cpu())
            all_scores.append(scores.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if not all_boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return boxes, scores, labels


def _classwise_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    iou: float,
    max_dets: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores, labels

    keep_all: List[torch.Tensor] = []
    for c in labels.unique():
        m = labels == c
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        k = nms(boxes[idx], scores[idx], float(iou))
        keep_all.append(idx[k])
    if not keep_all:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )
    keep = torch.cat(keep_all, dim=0)
    # global top-k by score
    keep = keep[torch.argsort(scores[keep], descending=True)]
    if int(max_dets) > 0 and keep.numel() > int(max_dets):
        keep = keep[: int(max_dets)]
    return boxes[keep], scores[keep], labels[keep]


def main() -> None:
    cli = _parse_args()
    root = _repo_root()
    os.chdir(str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Build args like main.py does, but allow overwriting existing argparse keys.
    from util.slconfig import SLConfig
    from main import get_args_parser as _get_main_args_parser
    import datasets

    parser = _get_main_args_parser()
    args = parser.parse_args([])
    args.config_file = str(cli.config_file)
    args.resume = str(cli.resume)
    args.eval = True
    args.amp = bool(cli.amp)
    args.device = str(cli.device)
    args.num_workers = int(cli.num_workers)
    args.seed = int(cli.seed)

    # dataset paths
    if cli.val_img_folder is not None:
        args.val_img_folder = cli.val_img_folder
    if cli.val_ann_file is not None:
        args.val_ann_file = cli.val_ann_file
    if cli.dataset_json_root:
        args.dataset_json_root = cli.dataset_json_root
        args.coco_path = ""
    if cli.dataset_images:
        args.dataset_images = cli.dataset_images
    if cli.coco_path:
        args.coco_path = cli.coco_path

    cfg = SLConfig.fromfile(args.config_file)
    # merge extra options (key=value)
    if cli.extra_options:
        from util.slconfig import DictAction

        # reuse DictAction parser to interpret "a=b" types
        opt_parser = argparse.ArgumentParser(add_help=False)
        opt_parser.add_argument("--options", nargs="+", action=DictAction)
        opt_ns = opt_parser.parse_args(["--options"] + list(cli.extra_options))
        cfg.merge_from_dict(opt_ns.options)

    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        setattr(args, k, v)

    # Determinism-ish
    torch.manual_seed(args.seed)
    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Build model
    from main import build_model_main

    model, criterion, postprocessors = build_model_main(args)
    model.to(device)
    model.eval()

    # Load checkpoint weights
    ckpt = torch_load_trusted(args.resume, map_location="cpu")
    sd = ckpt.get("ema_model", None) if ("ema_model" in ckpt and "checkpoint_best_ema" in str(args.resume)) else ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[tta_eval] load_state_dict strict=False:",
          "missing", len(missing), "unexpected", len(unexpected))

    # Dataset / evaluator
    dataset_val = datasets.build_dataset(image_set="val", args=args)
    base_ds = datasets.get_coco_api_from_dataset(dataset_val)
    coco_evaluator = CocoEvaluator(base_ds, ["bbox"])

    scales = _parse_scales(cli.scales)
    print("[tta_eval] scales:", scales, "max_size:", cli.max_size, "nms_iou:", cli.nms_iou, "max_dets:", cli.max_dets)

    predictions = {}
    for idx in range(len(dataset_val)):
        try:
            img_pil, target = dataset_val._get_prepared(idx)
        except Exception:
            # fallback to normal __getitem__ (but then TTA isn't correct)
            img_tensor, target = dataset_val[idx]
            raise RuntimeError("Dataset does not support _get_prepared; cannot run TTA.") from None

        if getattr(img_pil, "mode", None) != "RGB":
            img_pil = img_pil.convert("RGB")

        image_id = target["image_id"]
        if isinstance(image_id, torch.Tensor):
            image_id = int(image_id.item())
        else:
            image_id = int(image_id)

        boxes, scores, labels = _tta_predict_one(
            model,
            postprocessors,
            img_pil,
            device,
            scales=scales,
            max_size=int(cli.max_size),
            amp=bool(cli.amp),
        )

        boxes, scores, labels = _classwise_nms(
            boxes, scores, labels, iou=float(cli.nms_iou), max_dets=int(cli.max_dets)
        )

        predictions[image_id] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

        if (idx + 1) % 50 == 0:
            print(f"[tta_eval] processed {idx+1}/{len(dataset_val)}")

    coco_evaluator.update(predictions)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # Save raw predictions if requested via output_dir (reuse main.py convention if present)
    out_dir = getattr(args, "output_dir", "") or ""
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            out_p = os.path.join(out_dir, "tta_predictions.json")
            # lightweight json (only lists)
            dump = {
                str(k): {
                    "boxes": v["boxes"].tolist(),
                    "scores": v["scores"].tolist(),
                    "labels": v["labels"].tolist(),
                }
                for k, v in predictions.items()
            }
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(dump, f)
            print("[tta_eval] wrote:", out_p)
        except Exception:
            pass


if __name__ == "__main__":
    main()

