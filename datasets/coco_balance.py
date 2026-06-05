"""按图像级权重缓解 VisDrone 类不平衡；可与 recall_aps.pdf 建议的「小目标/长尾召回」融合。"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from datasets.coco import CocoDetection


def compute_train_image_weights_coco(
    dataset: "CocoDetection",
    *,
    small_area_max: Optional[float] = None,
    small_boost_scale: float = 0.0,
) -> List[float]:
    """
    含稀有类实例越多的图权重越大（sqrt 平滑）；可选再放大「小目标占比高」的图像，利于 Recall/APs。
    small_area_max: 如 1024 (=32×32)，按 COCO bbox 像素面积 w*h 统计小实例。
    small_boost_scale: >0 时 w *= (1 + scale * n_small / n_all)。
    """
    coco = dataset.coco
    ann_list = coco.dataset.get("annotations") or []
    cat_counts: dict[int, int] = {}
    for a in ann_list:
        cid = int(a["category_id"])
        cat_counts[cid] = cat_counts.get(cid, 0) + 1

    use_small = small_area_max is not None and float(small_boost_scale) > 0
    sam = float(small_area_max) if small_area_max is not None else 0.0
    sbs = float(small_boost_scale)

    weights: List[float] = []
    for img_id in dataset.ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if not ann_ids:
            weights.append(1.0)
            continue
        anns = coco.loadAnns(ann_ids)
        w = 1.0
        n_small = 0
        for a in anns:
            cid = int(a["category_id"])
            cnt = max(1, cat_counts.get(cid, 1))
            w = max(w, 1.0 / math.sqrt(float(cnt)))
            if use_small:
                _x, _y, bw, bh = a["bbox"]
                if float(bw) * float(bh) <= sam:
                    n_small += 1
        if use_small:
            n_all = len(anns)
            if n_all > 0:
                w *= 1.0 + sbs * (float(n_small) / float(n_all))
        weights.append(float(w))
    return weights
