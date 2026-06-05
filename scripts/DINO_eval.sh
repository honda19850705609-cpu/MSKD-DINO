#!/usr/bin/env bash
set -euo pipefail

coco_path=${1:?Usage: bash scripts/DINO_eval.sh /path/to/coco_or_visdrone_coco checkpoint.pth}
checkpoint=${2:?Usage: bash scripts/DINO_eval.sh /path/to/coco_or_visdrone_coco checkpoint.pth}
config_file=${3:-config/DINO/DINO_5scale_visdrone_final.py}
output_dir=${4:-logs/DINO/MSKD-VisDrone-eval}

python main.py \
	--output_dir "$output_dir" \
	-c "$config_file" --coco_path "$coco_path" \
	--eval --resume "$checkpoint" \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=True \
	dn_box_noise_scale=1.0
