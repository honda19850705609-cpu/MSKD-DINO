#!/usr/bin/env bash
set -euo pipefail

coco_path=${1:?Usage: bash scripts/DINO_train.sh /path/to/coco_or_visdrone_coco}
config_file=${2:-config/DINO/DINO_5scale_visdrone_final.py}
output_dir=${3:-logs/DINO/MSKD-VisDrone}

python main.py \
	--output_dir "$output_dir" -c "$config_file" --coco_path "$coco_path" \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=True \
	dn_box_noise_scale=1.0
