#!/usr/bin/env python3
"""
在验证集图片上绘制 DINO 预测框并保存为 PNG（便于肉眼看效果）。

不写 --checkpoint / --out_dir 时，默认使用 util.thesis_spec 中的主实验目录
（DEFAULT_FULL_TRAIN_OUTPUT_DIR，可用环境变量 DINO_TRAIN_OUTPUT_DIR 覆盖），
权重优先 checkpoint_best_regular.pth，输出到 <该目录>/pred_vis/。

示例:
  python tools/visdrone_draw_predictions.py --max_images 12 --score_thresh 0.15

  python tools/visdrone_draw_predictions.py ^
    --checkpoint F:/paper/VisDrone_processing/.../checkpoint.pth ^
    --out_dir F:/paper/VisDrone_processing/.../pred_vis

  # Colab 训练的权重在本机出图（覆盖 checkpoint 里的 /content/drive/... 路径）:
  python tools/visdrone_draw_predictions.py ^
    --checkpoint G:/.../checkpoint_best_regular.pth ^
    --out_dir G:/.../model_visualizations ^
    --data_root G:/.../VisDrone2019_DET_val_processed --max_images 20

  # 随机 8 张 + 叠真值框（看预测是否贴目标）:
  python tools/visdrone_draw_predictions.py --random_sample --draw_gt --max_images 8 ^
    --score_thresh 0.15 --data_root G:/.../你的COCO包 --train_result_dir G:/.../训练输出目录
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from datasets import build_dataset
from main import build_model_main
from util import misc as utils
from util.box_ops import box_cxcywh_to_xyxy
from util.thesis_spec import (
    VISDRONE_CATEGORY_NAMES,
    default_checkpoint_under_train_dir,
    resolved_train_result_dir,
)


def apply_visdrone_processed_root(args, root: str) -> None:
    """
    将 COCO 布局根目录（含 images/、annotations/instances_*.json）写回 args，
    用于 checkpoint 里仍是 Colab 路径时在本机 Windows 上可视化。
    """
    r = Path(root).expanduser().resolve()
    img = r / "images"
    ann_train = r / "annotations" / "instances_train2017.json"
    ann_val = r / "annotations" / "instances_val2017.json"
    if not img.is_dir():
        raise SystemExit(f'--data_root 下未找到 images 目录: {img}')
    if not ann_val.is_file():
        raise SystemExit(f'--data_root 下未找到验证标注: {ann_val}')
    args.coco_path = ""
    args.dataset_json_root = str(r)
    args.dataset_images = str(img)
    args.train_img_folder = str(img)
    args.val_img_folder = str(img)
    args.val_ann_file = str(ann_val)
    args.train_ann_file = str(ann_train if ann_train.is_file() else ann_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default='', help='checkpoint 路径；省略则从主实验目录自动查找')
    ap.add_argument('--out_dir', type=str, default='', help='输出目录；省略则为 <结果目录>/pred_vis')
    ap.add_argument('--train_result_dir', type=str, default='', help='训练输出根目录（找 checkpoint / 默认 out_dir）')
    ap.add_argument('--max_images', type=int, default=10)
    ap.add_argument(
        '--random_sample',
        action='store_true',
        help='从验证集中随机抽样（默认取前 max_images 张）',
    )
    ap.add_argument('--seed', type=int, default=42, help='--random_sample 时的随机种子')
    ap.add_argument(
        '--draw_gt',
        action='store_true',
        help='叠加绿色虚线真值框（便于对比预测是否贴目标）',
    )
    ap.add_argument('--score_thresh', type=float, default=0.25, help='置信度阈值（模型较弱时可降到 0.05）')
    ap.add_argument('--max_boxes', type=int, default=40, help='每张图最多绘制多少个框（按分数从高到低）')
    ap.add_argument('--device', type=str, default=None, help='cuda / cpu，默认自动')
    ap.add_argument(
        '--nonstrict_load',
        action='store_true',
        help='load_state_dict(strict=False)，用于与当前代码结构略不一致的旧 checkpoint',
    )
    ap.add_argument(
        '--data_root',
        type=str,
        default='',
        help='VisDrone 转 COCO 后的根目录（含 images/ 与 annotations/）；覆盖 checkpoint 内 Colab 路径，便于本机出图',
    )
    args_cli = ap.parse_args()

    result_root = resolved_train_result_dir(args_cli.train_result_dir)
    ckpt_path = (args_cli.checkpoint or '').strip()
    ckpt_auto = False
    if not ckpt_path:
        ckpt_auto = True
        cand = default_checkpoint_under_train_dir(result_root)
        if not cand:
            raise SystemExit(
                f'未找到 checkpoint，请指定 --checkpoint。已查找目录: {result_root}'
            )
        ckpt_path = str(cand)
    out_dir = (args_cli.out_dir or '').strip()
    if not out_dir:
        base = result_root if ckpt_auto else Path(ckpt_path).resolve().parent
        out_dir = str(base / 'pred_vis')

    ckpt = utils.torch_load_trusted(ckpt_path, map_location='cpu')
    if 'model' not in ckpt:
        raise SystemExit('checkpoint 中无 model 键')
    args = ckpt['args']
    if args_cli.device:
        args.device = args_cli.device
    else:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dr = (args_cli.data_root or '').strip()
    if dr:
        apply_visdrone_processed_root(args, dr)

    device = torch.device(args.device)
    model, criterion, postprocessors = build_model_main(args)
    sd = utils.adapt_dino_checkpoint_state_dict(model, ckpt['model'])
    model.load_state_dict(sd, strict=not args_cli.nonstrict_load)
    model.to(device)
    model.eval()

    dataset_val = build_dataset(image_set='val', args=args)
    os.makedirs(out_dir, exist_ok=True)

    post = postprocessors['bbox']
    n_total = len(dataset_val)
    n = min(n_total, args_cli.max_images)
    if args_cli.random_sample:
        rng = random.Random(args_cli.seed)
        indices = rng.sample(range(n_total), n)
    else:
        indices = list(range(n))
    print(f'checkpoint: {ckpt_path}')
    print(f'验证集共 {n_total} 张，将保存 {n} 张到: {out_dir}')
    if args_cli.random_sample:
        print(f'  随机抽样 seed={args_cli.seed}，indices={indices}')

    for j, i in enumerate(indices):
        img_t, target = dataset_val[i]
        img_b = img_t.unsqueeze(0).to(device)
        orig_size = target['orig_size'].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_b)
        res = post(outputs, orig_size)[0]

        scores = res['scores'].cpu().numpy()
        labels = res['labels'].cpu().numpy()
        boxes = res['boxes'].cpu().numpy()
        m = scores >= args_cli.score_thresh
        scores, labels, boxes = scores[m], labels[m], boxes[m]
        order = np.argsort(-scores)
        order = order[: args_cli.max_boxes]
        scores, labels, boxes = scores[order], labels[order], boxes[order]

        # 读原图（与标注 file_name 一致）
        coco = dataset_val.coco
        img_id = int(target['image_id'])
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(dataset_val.root, img_info['file_name'])
        if not os.path.isfile(img_path):
            img_path = os.path.join(str(dataset_val.root), os.path.basename(img_info['file_name']))
        pil = Image.open(img_path).convert('RGB')
        vis = pil.copy()
        draw = ImageDraw.Draw(vis)
        W, H = vis.size
        try:
            font = ImageFont.truetype('arial.ttf', 14)
        except OSError:
            font = ImageFont.load_default()

        if args_cli.draw_gt and target['boxes'].numel() > 0:
            oh, ow = int(target['orig_size'][0]), int(target['orig_size'][1])
            tb = target['boxes'].float().cpu()
            scale = torch.tensor([ow, oh, ow, oh], dtype=tb.dtype)
            tb_abs = tb * scale
            tb_xyxy = box_cxcywh_to_xyxy(tb_abs)
            glabels = target['labels'].cpu().numpy()
            for gi in range(tb_xyxy.shape[0]):
                gx1, gy1, gx2, gy2 = [int(round(x)) for x in tb_xyxy[gi].tolist()]
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(W - 1, gx2), min(H - 1, gy2)
                gcls = int(glabels[gi])
                # COCO category_id 常为 1..10；训练 target 也可能已是 0..9
                if 1 <= gcls <= len(VISDRONE_CATEGORY_NAMES):
                    gname = VISDRONE_CATEGORY_NAMES[gcls - 1]
                elif 0 <= gcls < len(VISDRONE_CATEGORY_NAMES):
                    gname = VISDRONE_CATEGORY_NAMES[gcls]
                else:
                    gname = f'id{gcls}'
                for off in range(3):
                    draw.rectangle(
                        [gx1 - off, gy1 - off, gx2 + off, gy2 + off],
                        outline=(0, 160, 0),
                        width=1,
                    )
                draw.text((gx1, max(2, gy1 - 28)), f'GT {gname}', fill=(0, 140, 0), font=font)

        kept = 0
        for sc, lb, box in zip(scores, labels, boxes):
            kept += 1
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            color = (
                int(37 + (lb * 47) % 200),
                int(67 + (lb * 91) % 180),
                int(97 + (lb * 13) % 200),
            )
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            name = VISDRONE_CATEGORY_NAMES[int(lb)] if 0 <= int(lb) < len(VISDRONE_CATEGORY_NAMES) else str(int(lb))
            txt = f'{name} {sc:.2f}'
            ty = max(2, y1 - 16)
            draw.text((x1, ty), txt, fill=color, font=font)

        out_name = f'pred_{j:03d}_idx{i}_id{img_id}_n{kept}.png'
        vis.save(os.path.join(out_dir, out_name))
        print(f'  [{j}] {out_name}  boxes>thresh={kept}  file={os.path.basename(img_path)}')

    print('完成。')


if __name__ == '__main__':
    main()
