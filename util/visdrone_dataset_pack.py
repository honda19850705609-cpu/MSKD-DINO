# VisDrone 自包含数据包（dataset_manifest.json + images/ + annotations/）
# 与标准 COCO 目录（train2017/val2017/）区分：见 main.normalize_visdrone_data_folder
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


def _read_manifest(root: Path) -> Optional[dict]:
    man = root / "dataset_manifest.json"
    if not man.is_file():
        return None
    try:
        return json.loads(man.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def resolve_dataset_json_root_and_images(args: Any) -> None:
    """
    解析 --dataset_json_root / DINO_DATASET_JSON_ROOT / DINO_DATASET_PACK_ROOT：
    - 若存在 dataset_manifest.json，按其中 paths_relative_to_pack 设置标注与图像目录；
    - 否则默认 annotations/instances_train2017.json、instances_val2017.json；
    - 若未指定 --dataset_images，且包内存在 images/，则 train/val 共用该目录（与 manifest 中
      train_val_share_same_image_folder 一致）。
    """
    json_root = (getattr(args, "dataset_json_root", None) or "").strip()
    if not json_root:
        json_root = (os.environ.get("DINO_DATASET_JSON_ROOT") or "").strip()
    if not json_root:
        json_root = (os.environ.get("DINO_DATASET_PACK_ROOT") or "").strip()
    if not json_root:
        return

    root = Path(json_root).expanduser().resolve()
    args.dataset_json_root = str(root)
    args.coco_path = ""

    data = _read_manifest(root)
    rel = (data or {}).get("paths_relative_to_pack") or {}

    def _p(key: str, default: str) -> Path:
        sub = rel.get(key) or default
        return root / sub

    train_ann = _p("instances_train2017", "annotations/instances_train2017.json")
    val_ann = _p("instances_val2017", "annotations/instances_val2017.json")
    if train_ann.is_file():
        args.train_ann_file = str(train_ann)
    if val_ann.is_file():
        args.val_ann_file = str(val_ann)

    img_rel = rel.get("images") or "images"
    pack_images = root / img_rel
    share = (data or {}).get("train_val_share_same_image_folder", True)

    img_dir_cli = (getattr(args, "dataset_images", None) or "").strip()
    if not img_dir_cli:
        img_dir_cli = (os.environ.get("DINO_DATASET_IMAGES") or "").strip()
    if img_dir_cli:
        d = str(Path(img_dir_cli).expanduser().resolve())
        if getattr(args, "train_img_folder", None) is None:
            args.train_img_folder = d
        if getattr(args, "val_img_folder", None) is None:
            args.val_img_folder = d
    elif pack_images.is_dir() and share:
        if getattr(args, "train_img_folder", None) is None:
            args.train_img_folder = str(pack_images)
        if getattr(args, "val_img_folder", None) is None:
            args.val_img_folder = str(pack_images)


def is_visdrone_self_contained_pack(root: Path) -> bool:
    """含 COCO 标注与 images/，但无 train2017/ 目录（自包含包）。"""
    if not root.is_dir():
        return False
    ann = root / "annotations" / "instances_train2017.json"
    if not ann.is_file():
        return False
    if (root / "train2017").is_dir():
        return False
    return (root / "images").is_dir()
