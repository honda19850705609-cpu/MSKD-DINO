"""COCO 标注与 DETR/DINO num_classes 一致性检查（避免 silent AP=0）。"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional, Set


def collect_annotation_category_ids(ann_path: str | Path, max_ann: Optional[int] = None) -> Set[int]:
    p = Path(ann_path)
    with p.open(encoding='utf-8') as f:
        data = json.load(f)
    anns = data.get('annotations') or []
    if max_ann is not None:
        anns = anns[:max_ann]
    return {int(a['category_id']) for a in anns if 'category_id' in a}


def validate_coco_ann_for_detr(
    ann_path: str | Path,
    num_classes: int,
    *,
    split_name: str = 'val',
) -> None:
    """
    DINO 期望：标注里的 category_id 为 [0, num_classes-1] 的整数，与 pred logits 维度和 PostProcess 输出一致。
    若 JSON 使用 1..N 或 COCO 原始 1..90，会导致训练/评测错位或 AP 接近 0。
    """
    p = Path(ann_path)
    if not p.is_file():
        return
    cat_ids = collect_annotation_category_ids(p)
    if not cat_ids:
        warnings.warn(f'[{split_name}] {p.name} 中未读到任何 category_id。')
        return
    lo, hi = min(cat_ids), max(cat_ids)
    msg_lines = []
    if lo < 0 or hi >= num_classes:
        msg_lines.append(
            f'[{split_name}] category_id 范围 [{lo}, {hi}] 与 num_classes={num_classes} 不兼容 '
            f'（需要所有 id 落在 [0, {num_classes - 1}]）。请改转换脚本或调整 num_classes。'
        )
    expected = set(range(num_classes))
    if not cat_ids.issubset(expected):
        msg_lines.append(
            f'[{split_name}] 标注中出现了 num_classes 之外的 id: {sorted(cat_ids - expected)[:15]}'
        )
    missing_in_data = expected - cat_ids
    if missing_in_data:
        warnings.warn(
            f'[{split_name}] 下列 category_id 在标注中未出现: {sorted(missing_in_data)}',
            stacklevel=2,
        )
    if msg_lines:
        raise ValueError('\n'.join(msg_lines))


def infer_num_classes_from_coco_json(ann_path: str | Path) -> Optional[int]:
    """
    从 COCO JSON 的 categories[].id 推断 DINO num_classes（= max(category_id)+1，
    与 models/dino/dino.py 中 build 注释一致：VisDrone 为 0..9 -> 10，COCO 为 1..90 -> 91）。
    """
    p = Path(ann_path)
    if not p.is_file():
        return None
    try:
        with p.open(encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    cats = data.get('categories') or []
    if not cats:
        return None
    ids = [int(c['id']) for c in cats if isinstance(c, dict) and 'id' in c]
    if not ids:
        return None
    return max(ids) + 1


def resolve_ann_path_for_num_classes_infer(args) -> Optional[str]:
    """优先 val 标注（验证常用），否则 train；标准 coco_path 下读 annotations/instances_*.json。"""
    for key in ('val_ann_file', 'train_ann_file'):
        p = getattr(args, key, None)
        if p and Path(p).is_file():
            return str(Path(p).expanduser().resolve())
    coco = (getattr(args, 'coco_path', None) or '').strip()
    if coco:
        root = Path(coco).expanduser().resolve()
        for name in ('instances_val2017.json', 'instances_train2017.json'):
            cand = root / 'annotations' / name
            if cand.is_file():
                return str(cand)
    return None


def maybe_align_num_classes_from_dataset(args) -> None:
    """
    默认配置为 COCO 91 类时，若标注实为 N 类（如 VisDrone 10 类），自动写入
    args.options 中的 num_classes / dn_labelbook_size，避免与 10 类 checkpoint 的 class_embed 不匹配。
    """
    if getattr(args, 'no_auto_num_classes', False):
        return
    if getattr(args, 'thesis_visdrone_50', False):
        return
    opt = getattr(args, 'options', None)
    if opt is not None and 'num_classes' in opt:
        return
    ann = resolve_ann_path_for_num_classes_infer(args)
    if not ann:
        return
    n = infer_num_classes_from_coco_json(ann)
    if n is None or n == 91:
        return
    if args.options is None:
        args.options = {}
    if 'num_classes' not in args.options:
        args.options['num_classes'] = n
    if 'dn_labelbook_size' not in args.options:
        args.options['dn_labelbook_size'] = n
    print(
        f'[dataset] 根据标注「{Path(ann).name}」推断 num_classes={n}（dn_labelbook_size={n}），'
        '已合并到 --options，与 COCO 默认 91 类区分。若需手动指定请用 --options num_classes=…；'
        '关闭自动推断请加 --no_auto_num_classes。'
    )
