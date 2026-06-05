"""任务书 headline 指标：mAP@0.5、APs、各类 AP@0.5（便于论文表格）。"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from util.thesis_spec import CORE_CLASS_INDICES_TASKBOOK, VISDRONE_CATEGORY_NAMES


def _ap_per_category_at_iou(coco_eval, iou_thr: float = 0.5) -> list[tuple[int, str, float]]:
    """Return [(cat_id, name, ap50), ...] for params.catIds; name from VisDrone list when len matches."""
    if coco_eval.eval is None or "precision" not in coco_eval.eval:
        return []
    prec = coco_eval.eval["precision"]
    iou_thrs = coco_eval.params.iouThrs
    ixs = np.where(np.isclose(iou_thrs, iou_thr))[0]
    if len(ixs) == 0:
        return []
    i0 = int(ixs[0])
    area_idx = 0
    maxdet_idx = prec.shape[4] - 1
    cat_ids = list(coco_eval.params.catIds)
    out = []
    for ci, cid in enumerate(cat_ids):
        s = prec[i0, :, ci, area_idx, maxdet_idx]
        s = s[s > -1]
        ap = float(np.mean(s)) if s.size else float("nan")
        name = (
            VISDRONE_CATEGORY_NAMES[cid]
            if 0 <= cid < len(VISDRONE_CATEGORY_NAMES)
            else f"id_{cid}"
        )
        out.append((int(cid), name, ap))
    return out


def log_thesis_coco_bbox_metrics(
    coco_eval_bbox: Any,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    pycocotools COCOeval bbox: stats[0]=mAP, [1]=mAP@0.5, [2]=mAP@0.75,
    [3]=AP small, [4]=medium, [5]=large (COCO 标准 area 划分).
    """
    log = logger.info if logger else print
    stats = list(coco_eval_bbox.stats)
    if len(stats) < 6:
        log(f"[任务书指标] coco bbox stats 长度异常: {len(stats)}")
        return
    map50 = stats[1]
    aps = stats[3]
    log(
        "[任务书 headline] "
        f"mAP@0.5={map50:.4f} | AP_small={aps:.4f} "
        f"(另: mAP@[.5:.95]={stats[0]:.4f}, mAP@0.75={stats[2]:.4f})"
    )
    per_cat = _ap_per_category_at_iou(coco_eval_bbox, 0.5)
    if not per_cat:
        return
    core = [x for x in per_cat if x[0] in CORE_CLASS_INDICES_TASKBOOK]
    if core:
        mean_core = float(np.nanmean([x[2] for x in core]))
        log(
            f"[任务书 核心类 AP@0.5 均值] categories={tuple(x[1] for x in core)} -> mean={mean_core:.4f}"
        )
    lines = " | ".join(f"{n}:{ap:.3f}" for _, n, ap in per_cat)
    log(f"[各类 AP@0.5] {lines}")
    log(
        "[任务书说明] FPS = 1 / 单张平均推理时间；可用纯推理脚本单独统计；"
        "Precision/Recall/F1、FN/FP Rate 需在验证集上做 per-image 统计后汇总。"
    )
