# VisDrone 结构性升级：P2 + 5-level features +（可选）BiFPN
# 目标：优先提升 AP_small 与 Recall（密集小目标场景）
#
# 用法示例：
#   python main.py -c config/DINO/DINO_5scale_visdrone_structural.py --amp ...
#
# 提示：P2 会显著增加显存；若 OOM，把 batch_size 调小，或设 p2_proj_stride=2。
_base_ = ['DINO_4scale_visdrone_stageA_target_50ep.py']

# --- P2 + 5-scale ---
return_interm_indices = [0, 1, 2, 3]   # include layer1(P2)
num_feature_levels = 5                 # P2,P3,P4,P5 + P6

# 显存压力大时建议启用：把 P2 在 input_proj 时 stride=2 先降采样（仍保留 layer1 语义）
p2_proj_stride = 2

# --- Optional BiFPN neck ---
use_bifpn_neck = True
bifpn_repeats = 2

# 结构升级后建议更保守的 batch（按你 GPU 调；H100 60G 通常 batch=2~4）
batch_size = 2

# Recall 优先：提高分类损失权重，并把 focal_alpha 提到 0.5（更“大胆”）
cls_loss_coef = 2.0
focal_alpha = 0.5

# Long-tail: per-class focal weights (positive targets only)
use_visdrone_class_weights = True
visdrone_class_weights = [1.0, 1.5, 3.0, 1.0, 2.5, 2.5, 3.5, 8.0, 5.0, 1.5]

# 评测/输出：密集场景保持更高的 maxDets（代码里已改到500），这里让 topk 更充足
num_select = 500

