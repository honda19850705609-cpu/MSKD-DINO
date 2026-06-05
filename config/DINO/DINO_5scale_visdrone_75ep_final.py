# ==============================================================
# DINO_5scale_visdrone_75ep_final.py
# 75 轮最终训练：冲争取目标上限，为后续轻量化留余量
#
# 核心改动 vs 当前 stageA_target_50ep：
#   损失: cls 1.0→3.0, alpha 0.25→0.6, giou 2.5→2.0
#   架构: 4scale→5scale (P2层), BiFPN 双向融合
#   增强: crop [384,800]→[600,1200], max_size 1400→2000
#   类别: 开启加权, awning-tricycle ×7
#   训练: 50ep→75ep, LR 衰减推迟到 [58,70]
# ==============================================================

# ==================== 数据增强 ====================
data_aug_scales = [
    480, 512, 544, 576, 608, 640, 672, 704, 736, 768,
    800, 864, 928, 1000, 1100, 1200, 1400, 1500,
]
data_aug_max_size = 2000
data_aug_scales2_resize = [600, 800, 1000, 1200, 1500]
data_aug_scales2_crop = [600, 1200]
data_aug_scale_overlap = None

# ==================== 模型架构 ====================
num_classes = 10
modelname = 'dino'
backbone = 'resnet50'
use_checkpoint = False
dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20

# P2 高分辨率特征层
return_interm_indices = [0, 1, 2, 3]   # 含 layer1 (P2, stride=4, 256ch)
num_feature_levels = 5                  # P2 + P3 + P4 + P5 + P6
p2_proj_stride = 2                      # P2 用 stride=2 卷积降采样

# BiFPN
use_bifpn_neck = True
bifpn_repeats = 2

backbone_freeze_keywords = None

# Transformer
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = True
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True

# ==================== 学习率 ====================
lr = 0.0002
param_dict_type = 'default'
lr_backbone = 2e-5
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
lr_encoder_mult = 1.3
ddetr_lr_param = False
weight_decay = 0.0001
clip_max_norm = 0.1

# ==================== 训练轮次 ====================
batch_size = 4
epochs = 75
lr_drop = 999                  # 不用单点衰减
onecyclelr = False
multi_step_lr = True
lr_drop_list = [58, 70]        # 75 轮中在第 58 和 70 轮衰减
lr_warmup_epochs = 5
save_checkpoint_interval = 1

# ==================== 损失函数 ====================
cls_loss_coef = 3.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
box_iou_loss_type = 'siou'
focal_alpha = 0.6
mask_loss_coef = 1.0
dice_loss_coef = 1.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False

# 匹配成本
set_cost_class = 3.0
set_cost_bbox = 5.0
set_cost_giou = 2.0

# ==================== 匹配与解码 ====================
decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1
dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True
match_unstable_error = True
use_detached_boxes_dec_out = False

# ==================== DN ====================
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.5
dn_label_noise_ratio = 0.55
embed_init_tgt = True
dn_labelbook_size = 10

# ==================== EMA ====================
use_ema = True
ema_decay = 0.9997
ema_epoch = 0

# ==================== 类别加权 ====================
use_visdrone_class_weights = True
visdrone_class_weights = [
    1.0,   # 0 pedestrian  (AP50=0.65, TP=960, FN=720)
    1.8,   # 1 people      (AP50=0.53, TP=500, FN=520 → 漏检多)
    4.0,   # 2 bicycle     (AP50=0.39, R=0.34 → 严重漏检)
    0.6,   # 3 car         (AP50=0.89, R=0.84 → 已饱和)
    2.5,   # 4 van         (AP50=0.58, R=0.53)
    3.5,   # 5 truck       (AP50=0.44, R=0.35 → 严重漏检)
    3.0,   # 6 tricycle    (AP50=0.43, R=0.38)
    7.0,   # 7 awning-tri  (AP50=0.21, R=0.22 → 最差)
    2.0,   # 8 bus         (AP50=0.70, R=0.61)
    1.0,   # 9 motor       (AP50=0.72, R=0.66)
]

# ==================== 数据增强开关 ====================
copy_paste_prob = 0.8
copy_paste_max_objects = 6
copy_paste_max_src_area_pixels = 2048
train_color_jitter = True
train_motion_blur_prob = 0.2
train_motion_blur_kernel = 11
train_class_balance_sampler = True
train_balance_small_area_max = 1024
train_balance_small_boost_scale = 0.7
train_gaussian_blur_prob = 0.08
train_gaussian_blur_radius = 0.8
train_mosaic_prob = 0.35
train_mosaic_size = (1280, 800)

# ==================== 杂项 ====================
frozen_weights = None
checkpoint_every_iters = 0
cuda_empty_cache_every_iters = 200
visdrone_retrain_plan_aug = False
laptop_stable_training = False
laptop_stable_num_workers_cap = 8

