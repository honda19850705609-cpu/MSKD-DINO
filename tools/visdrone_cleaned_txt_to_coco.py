#!/usr/bin/env python3
"""
Convert VisDrone-style *_cleaned.txt (normalized cx,cy,w,h + class id) + images to COCO JSON
for DINO training. Each line: ...,cx,cy,w,h,conf,cls,...

Example (与 util/thesis_spec.py 默认路径一致):
  python tools/visdrone_cleaned_txt_to_coco.py ^
    --images_dir F:/paper/VisDrone_out/VisDrone2019_train_out_random_50example_origin ^
    --txt_dir F:/paper/VisDrone_out/VisDrone2019_train_out_random_50example_processed_for_dino ^
    --out_root F:/paper/VisDrone_out/VisDrone2019_train_out_random_50example_processed_for_dino ^
    --val_ratio 0.2 --seed 42
  (若图片在子目录 images/ 下，把 --images_dir 改为 .../origin/images)
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image
VISDRONE_CATEGORIES = [
    {"id": i, "name": n, "supercategory": "none"}
    for i, n in enumerate([
        "pedestrian", "people", "bicycle", "car", "van", "truck",
        "tricycle", "awning-tricycle", "bus", "motor",
    ])
]


def parse_line(line):
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    if len(parts) < 8:
        return None
    cx, cy, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
    cls = int(parts[7])
    return cx, cy, w, h, cls


def line_to_bbox(cx, cy, w, h, img_w, img_h):
    x = (cx - w / 2.0) * img_w
    y = (cy - h / 2.0) * img_h
    bw = w * img_w
    bh = h * img_h
    x = max(0.0, min(x, img_w - 1))
    y = max(0.0, min(y, img_h - 1))
    bw = max(1.0, min(bw, img_w - x))
    bh = max(1.0, min(bh, img_h - y))
    return [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)]


def collect_pairs(images_dir, txt_dir, suffix="_cleaned.txt"):
    images_dir = Path(images_dir)
    txt_dir = Path(txt_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    txt_map = {}
    for p in txt_dir.rglob(f"*{suffix}"):
        if not p.is_file():
            continue
        st = p.name[: -len(suffix)] if p.name.endswith(suffix) else p.stem
        txt_map[st] = p
    pairs = []
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in exts:
            continue
        stem = img_path.stem
        cand = txt_map.get(stem)
        if cand is None:
            alt = [v for k, v in txt_map.items() if k.startswith(stem) or stem.startswith(k)]
            cand = alt[0] if len(alt) == 1 else None
        if cand is None:
            cand = txt_dir / f"{stem}{suffix}"
            if not cand.is_file():
                alt = list(txt_dir.glob(f"{stem}*{suffix}"))
                cand = alt[0] if alt else None
        if cand is None:
            continue
        pairs.append((img_path, cand))
    return pairs


def build_coco(images_ann_list, start_ann_id=1):
    images = []
    annotations = []
    ann_id = start_ann_id
    for img_id, (img_path, txt_path) in enumerate(images_ann_list, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        file_name = img_path.name
        images.append({"id": img_id, "width": w, "height": h, "file_name": file_name})
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = parse_line(line)
                if p is None:
                    continue
                cx, cy, bw_n, bh_n, cls = p
                if cls < 0 or cls > 9:
                    continue
                bbox = line_to_bbox(cx, cy, bw_n, bh_n, w, h)
                area = bbox[2] * bbox[3]
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": VISDRONE_CATEGORIES,
    }


def write_visdrone_coco_to_out_root(
    images_dir,
    txt_dir,
    out_root,
    val_ratio=0.2,
    seed=42,
    txt_suffix="_cleaned.txt",
):
    """
    Programmatic API: write instances_train2017.json / instances_val2017.json under out_root/annotations/.
    Returns (train_json_path, val_json_path) as Path objects.
    """
    images_dir = Path(images_dir)
    txt_dir = Path(txt_dir)
    out_root = Path(out_root)
    pairs = collect_pairs(images_dir, txt_dir, suffix=txt_suffix)
    if not pairs:
        raise ValueError(
            f"No image/txt pairs: images_dir={images_dir} txt_dir={txt_dir}. "
            "Need matching image stems and *_cleaned.txt."
        )
    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 1
    if len(pairs) <= 2:
        n_val = 1
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    ann_dir = out_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    train_json = build_coco(train_pairs)
    val_json = build_coco(val_pairs)
    train_path = ann_dir / "instances_train2017.json"
    val_path = ann_dir / "instances_val2017.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_json, f)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_json, f)
    return train_path.resolve(), val_path.resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True, help="Folder with source images (.jpg, ...)")
    ap.add_argument("--txt_dir", type=str, required=True, help="Folder with matching *_cleaned.txt files")
    ap.add_argument("--out_root", type=str, required=True, help="Output COCO root (annotations/ + optional train2017,val2017 symlinks info)")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--txt_suffix", type=str, default="_cleaned.txt")
    args = ap.parse_args()

    train_path, val_path = write_visdrone_coco_to_out_root(
        args.images_dir,
        args.txt_dir,
        args.out_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        txt_suffix=args.txt_suffix,
    )
    with open(train_path, encoding="utf-8") as f:
        train_json = json.load(f)
    with open(val_path, encoding="utf-8") as f:
        val_json = json.load(f)

    src_images = Path(args.images_dir).resolve()
    print(f"Wrote {train_path} ({len(train_json['images'])} images, {len(train_json['annotations'])} boxes)")
    print(f"Wrote {val_path} ({len(val_json['images'])} images, {len(val_json['annotations'])} boxes)")
    print(f"Image files remain in: {src_images}")
    print("\nOption A — same image folder for train/val (recommended on Windows):")
    print(
        '  python main.py -c config/DINO/DINO_4scale.py --output_dir "YOUR_OUTPUT_DIR" '
        f'--train_img_folder "{src_images}" --train_ann_file "{train_path}" '
        f'--val_img_folder "{src_images}" --val_ann_file "{val_path}" '
        "--options num_classes=10 dn_labelbook_size=10"
    )
    print("\nOption B — COCO layout: create folders train2017 and val2017 (copy or junction to images), set --coco_path to:")
    print(f'  "{out_root}"')


if __name__ == "__main__":
    main()
