# 与毕业论文任务书对齐的常量（VisDrone / DINO-DETR 简化方向）
from __future__ import annotations

import os
from pathlib import Path
# 任务书 PDF 可参考（本仓库不读取该文件，仅作记录）:
#   E:/normal/2025大四上/毕业论文/任务书/任务书-史宏达.pdf

TASKBOOK_TITLE = "基于 Transformer 的无人机密集目标检测改进（DINO-DETR 简化方向）"

# VisDrone-DET 10 类（与 tools/visdrone_cleaned_txt_to_coco.VISDRONE_CATEGORIES 一致）
VISDRONE_CATEGORY_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)

# 任务书「行人、车辆、自行车」核心类：行人 + 自行车 + 常见机动车类别
# pedestrian, people, bicycle, car, van, truck, bus, motor（不含 tricycle / awning-tricycle）
CORE_CLASS_INDICES_TASKBOOK = (0, 1, 2, 3, 4, 5, 8, 9)

# 数据流程默认目录（可通过环境变量 THESIS_IMAGES_DIR / THESIS_COCO_ROOT 或 main.py CLI 覆盖）
# 原始图片根目录（图片可直接在此目录，或在其子目录 images/ 下）
DEFAULT_THESIS_IMAGE_ROOT = r"F:\paper\VisDrone_out\VisDrone2019_train_out_random_50example_origin"
# *_cleaned.txt 与 COCO 输出根目录（运行转换脚本后会出现 annotations/instances_*.json）
DEFAULT_THESIS_PROCESSED_ROOT = r"F:\paper\VisDrone_out\VisDrone2019_train_out_random_50example_processed_for_dino"

DEFAULT_THESIS_TXT_DIR = DEFAULT_THESIS_PROCESSED_ROOT
DEFAULT_THESIS_COCO_ROOT = DEFAULT_THESIS_PROCESSED_ROOT
# 兼容旧名：默认解析后的图片文件夹（见 default_thesis_image_folder）
DEFAULT_THESIS_IMAGES_DIR = DEFAULT_THESIS_IMAGE_ROOT

# 当前默认训练/续训输出目录（含 checkpoint / log；可用环境变量 DINO_TRAIN_OUTPUT_DIR 覆盖）
DEFAULT_FULL_TRAIN_OUTPUT_DIR = r"F:\paper\VisDrone_processing\result_DETR_VisDrone2019_DET_test_dev_DINO"

# COCO 上训好的 DINO 全模权重 .pth（用于 VisDrone 等微调；留空则仅用命令行/环境变量/交互）
# 环境变量优先：DINO_PRETRAIN_MODEL_PATH 或 DINO_PRETRAIN
DEFAULT_DINO_COCO_PRETRAIN_PATH = r""


def resolved_train_result_dir(cli_dir: str = "") -> Path:
    """
    解析「训练结果根目录」：CLI > 环境变量 DINO_TRAIN_OUTPUT_DIR > DINO_OUTPUT_DIR >
    若 DEFAULT_FULL_TRAIN_OUTPUT_DIR 下存在 log.txt 则采用该路径。
    """
    v = (cli_dir or "").strip()
    if v:
        return Path(v).expanduser().resolve()
    for key in ("DINO_TRAIN_OUTPUT_DIR", "DINO_OUTPUT_DIR"):
        e = (os.environ.get(key) or "").strip()
        if e:
            return Path(e).expanduser().resolve()
    p = Path(DEFAULT_FULL_TRAIN_OUTPUT_DIR)
    if (p / "log.txt").is_file():
        return p.resolve()
    return p.expanduser().resolve()


def default_checkpoint_under_train_dir(
    root: Path | None = None,
    *,
    prefer_best_regular: bool = True,
) -> Path | None:
    """在训练输出目录下查找常用 checkpoint 文件名。"""
    root = root or resolved_train_result_dir()
    order = (
        ("checkpoint_best_regular.pth", "checkpoint.pth", "checkpoint_best_ema.pth")
        if prefer_best_regular
        else ("checkpoint.pth", "checkpoint_best_regular.pth", "checkpoint_best_ema.pth")
    )
    for name in order:
        cand = root / name
        if cand.is_file():
            return cand
    return None


def default_thesis_image_folder() -> str:
    """训练/转换用的图片目录：若根下无常见图片则尝试 根目录/images。"""
    root = Path(DEFAULT_THESIS_IMAGE_ROOT)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if root.is_dir():
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                return str(root)
        nested = root / "images"
        if nested.is_dir():
            return str(nested)
    return str(root)
