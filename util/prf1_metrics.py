from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class PRF1:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1, bx1[None, :])
    iy1 = np.maximum(ay1, by1[None, :])
    ix2 = np.minimum(ax2, bx2[None, :])
    iy2 = np.minimum(ay2, by2[None, :])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def _greedy_match_iou(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    *,
    iou_thr: float,
) -> Tuple[int, int, int]:
    """
    Greedy matching: each prediction matches at most one gt and vice versa.
    Returns (tp, fp, fn).
    """
    n_p = int(pred_boxes.shape[0])
    n_g = int(gt_boxes.shape[0])
    if n_p == 0 and n_g == 0:
        return 0, 0, 0
    if n_p == 0:
        return 0, 0, n_g
    if n_g == 0:
        return 0, n_p, 0
    iou = _iou_xyxy(pred_boxes, gt_boxes)
    matched_g = np.zeros((n_g,), dtype=bool)
    tp = 0
    for i in range(n_p):
        # find best gt for this pred among unmatched
        j = int(np.argmax(iou[i]))
        if matched_g[j]:
            continue
        if float(iou[i, j]) >= float(iou_thr):
            matched_g[j] = True
            tp += 1
    fp = n_p - tp
    fn = n_g - int(matched_g.sum())
    return tp, fp, fn


def compute_prf1_for_coco_like(
    *,
    gt_by_image: Dict[int, Dict[str, np.ndarray]],
    pred_by_image: Dict[int, Dict[str, np.ndarray]],
    score_thr: float,
    iou_thr: float = 0.5,
    class_ids: Sequence[int] | None = None,
) -> PRF1:
    """
    gt_by_image[img_id] = {"boxes": (G,4) xyxy, "labels": (G,) int}
    pred_by_image[img_id] = {"boxes": (P,4) xyxy, "labels": (P,) int, "scores": (P,) float}
    """
    tp = fp = fn = 0
    cls_set = set(class_ids) if class_ids is not None else None

    img_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    for img_id in img_ids:
        gt = gt_by_image.get(img_id, None)
        pr = pred_by_image.get(img_id, None)
        if gt is None:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)
        else:
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]
        if pr is None:
            pr_boxes = np.zeros((0, 4), dtype=np.float32)
            pr_labels = np.zeros((0,), dtype=np.int64)
            pr_scores = np.zeros((0,), dtype=np.float32)
        else:
            pr_boxes = pr["boxes"]
            pr_labels = pr["labels"]
            pr_scores = pr["scores"]

        # optional class subset
        if cls_set is not None:
            gmask = np.array([int(x) in cls_set for x in gt_labels], dtype=bool)
            pmask = np.array([int(x) in cls_set for x in pr_labels], dtype=bool)
            gt_boxes = gt_boxes[gmask]
            gt_labels = gt_labels[gmask]
            pr_boxes = pr_boxes[pmask]
            pr_labels = pr_labels[pmask]
            pr_scores = pr_scores[pmask]

        # score threshold
        keep = pr_scores >= float(score_thr)
        pr_boxes = pr_boxes[keep]
        pr_labels = pr_labels[keep]
        pr_scores = pr_scores[keep]

        # per-class matching (prevents cross-class matching inflating)
        classes_here = set(gt_labels.tolist()) | set(pr_labels.tolist())
        for c in classes_here:
            c = int(c)
            gcb = gt_boxes[gt_labels == c]
            pcb = pr_boxes[pr_labels == c]
            tpi, fpi, fni = _greedy_match_iou(pcb, gcb, iou_thr=iou_thr)
            tp += tpi
            fp += fpi
            fn += fni

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return PRF1(precision=precision, recall=recall, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn))


def find_best_f1_threshold(
    *,
    gt_by_image: Dict[int, Dict[str, np.ndarray]],
    pred_by_image: Dict[int, Dict[str, np.ndarray]],
    thresholds: Iterable[float],
    iou_thr: float = 0.5,
    class_ids: Sequence[int] | None = None,
) -> Tuple[float, PRF1]:
    best_thr = 0.0
    best = PRF1(precision=0.0, recall=0.0, f1=-1.0, tp=0, fp=0, fn=0)
    for t in thresholds:
        cur = compute_prf1_for_coco_like(
            gt_by_image=gt_by_image,
            pred_by_image=pred_by_image,
            score_thr=float(t),
            iou_thr=iou_thr,
            class_ids=class_ids,
        )
        if cur.f1 > best.f1:
            best = cur
            best_thr = float(t)
    return best_thr, best

