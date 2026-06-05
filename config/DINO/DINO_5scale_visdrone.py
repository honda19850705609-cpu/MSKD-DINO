# DINO VisDrone 5-scale（P2+P3+P4+P5+P6）+ BiFPN + 重训方案
# 对齐毕业论文目录《dino_visdrone_retrain_plan.4.8.md》：特征层、损失、采样与训练策略
#
# 用法示例：
#   python main.py -c config/DINO/DINO_5scale_visdrone.py --amp \
#     --train_img_folder ... --train_ann_file ... --val_img_folder ... --val_ann_file ...
#
# 显存：建议 batch_size=3（5070Ti 级可再减为 2）；OOM 时确认 p2_proj_stride=2（默认已开）
_base_ = ['DINO_5scale_visdrone_structural.py']

# --- 与方案文档 §5 对齐 ---
epochs = 50
lr = 2e-4
lr_backbone = 2e-5
multi_step_lr = True
lr_drop_list = [40]
lr_warmup_epochs = 3
batch_size = 3
num_select = 300

use_ema = True
ema_decay = 0.9997
ema_epoch = 0

# 验证集更大输入（利于小目标 eval）
data_aug_val_scales = [1000]
data_aug_val_max_size = 1667

# 方案专用增强（大尺度 / crop / 光度 / 模糊）；与 thesis 的 jitter+mosaic 二选一为主增强源
visdrone_retrain_plan_aug = True

# 继承 structural 的：return_interm_indices, num_feature_levels=5, p2_proj_stride=2,
# use_bifpn_neck, cls_loss_coef=2, focal_alpha=0.5, use_visdrone_class_weights, box_iou 见下层 thesis
