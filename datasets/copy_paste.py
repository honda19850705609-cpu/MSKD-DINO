# Simple Copy-Paste for detection (boxes only), after Ghiasi et al., CVPR 2021.
# Runs on PIL + xyxy pixel boxes before ToTensor; intended for VisDrone / crowded small objects.
from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
from PIL import Image


def _donor_candidate_indices(
    d_boxes: torch.Tensor,
    *,
    min_side: int,
    max_src_area_pixels: Optional[float],
) -> list[int]:
    """《DINO小目标检测优化》：优先只从「小目标」框选 patch（COCO 小目标面积 < 32×32）。"""
    n = d_boxes.shape[0]
    cand: list[int] = []
    for j in range(n):
        x1, y1, x2, y2 = [float(t) for t in d_boxes[j].tolist()]
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_side or bh < min_side:
            continue
        ar = bw * bh
        if max_src_area_pixels is not None and ar > float(max_src_area_pixels):
            continue
        cand.append(j)
    return cand


def apply_simple_copy_paste(
    img: Image.Image,
    target: dict,
    donor_img: Image.Image,
    donor_target: dict,
    *,
    max_objects: int = 3,
    min_side: int = 4,
    max_src_area_pixels: Optional[float] = None,
) -> Tuple[Image.Image, dict]:
    """
    Paste up to max_objects instances from donor onto img. Boxes/labels/area/iscrowd updated.
    max_src_area_pixels: 若设置，只从面积不超过该值的 donor 框中选取；若无候选则退回全部框。
    """
    if donor_target["boxes"].numel() == 0:
        return img, target

    W, H = img.size
    d_boxes = donor_target["boxes"].clone()
    d_labels = donor_target["labels"].clone()
    d_area = donor_target.get("area", None)
    d_crowd = donor_target.get("iscrowd", None)
    n = d_boxes.shape[0]
    cand = _donor_candidate_indices(d_boxes, min_side=min_side, max_src_area_pixels=max_src_area_pixels)
    if not cand:
        cand = list(range(n))
    k = random.randint(1, min(max_objects, len(cand)))
    pick = random.sample(cand, k)

    img = img.copy()
    boxes_out = [target["boxes"].clone()]
    labels_out = [target["labels"].clone()]
    area_out = [target["area"].clone()] if "area" in target else []
    crowd_out = [target["iscrowd"].clone()] if "iscrowd" in target else []

    for j in pick:
        box = d_boxes[j]
        x1, y1, x2, y2 = [int(round(float(t))) for t in box.tolist()]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(donor_img.size[0], x2)
        y2 = min(donor_img.size[1], y2)
        if x2 - x1 < min_side or y2 - y1 < min_side:
            continue
        patch = donor_img.crop((x1, y1, x2, y2))
        pw, ph = patch.size
        if pw < min_side or ph < min_side:
            continue
        max_x0 = max(0, W - pw)
        max_y0 = max(0, H - ph)
        if max_x0 < 0 or max_y0 < 0:
            continue
        x0 = random.randint(0, max_x0)
        y0 = random.randint(0, max_y0)
        new_box = torch.tensor(
            [float(x0), float(y0), float(x0 + pw), float(y0 + ph)],
            dtype=torch.float32,
        )
        # skip if pasted patch has tiny intersection with existing (optional overlap ok)
        img.paste(patch, (x0, y0))
        boxes_out.append(new_box.unsqueeze(0))
        labels_out.append(d_labels[j].view(1))
        if d_area is not None:
            area_out.append(torch.tensor([float(pw * ph)], dtype=torch.float32))
        if d_crowd is not None:
            crowd_out.append(d_crowd[j].view(1))

    target = target.copy()
    target["boxes"] = torch.cat(boxes_out, dim=0)
    target["labels"] = torch.cat(labels_out, dim=0)
    if area_out:
        target["area"] = torch.cat(area_out, dim=0)
    if crowd_out:
        target["iscrowd"] = torch.cat(crowd_out, dim=0)
    return img, target
