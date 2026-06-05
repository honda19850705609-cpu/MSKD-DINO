# ==============================================================
# DINO_5scale_visdrone_final.py
# 一步到位冲线：P2 + BiFPN + 高 cls_loss + 类别加权 + 激进增强
# 目标：AP50≥0.62, APs≥0.30, 核心F1≥0.82, 全类F1≥0.78
# ==============================================================

# ---- 数据增强 ----
data_aug_scales = [
    480, 512, 544, 576, 608, 640, 672, 704, 736, 768,
    800, 864, 928, 1000, 1100, 1200, 1400, 1500,
]
data_aug_max_size = 2000
data_aug_scales2_resize = [600, 800, 1000, 1200, 1500]
data_aug_scales2_crop = [600, 1200]
data_aug_scale_overlap = None

# ---- 模型架构 ----
num_classes = 10
modelname = 'dino'
backbone = 'resnet50'
use_checkpoint = False
dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [0, 1, 2, 3]
backbone_freeze_keywords = None
num_feature_levels = 5
p2_proj_stride = 2
use_bifpn_neck = True
bifpn_repeats = 2

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

# ---- 学习率 ----
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

# ---- 训练轮次 ----
batch_size = 4
epochs = 60
lr_drop = 999
onecyclelr = False
multi_step_lr = True
lr_drop_list = [45, 55]
lr_warmup_epochs = 5
save_checkpoint_interval = 1

# ---- 损失函数 ----
cls_loss_coef = 3.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
box_iou_loss_type = 'siou'
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.6
set_cost_class = 3.0
set_cost_bbox = 5.0
set_cost_giou = 2.0

# ---- 匹配与解码 ----
decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1
dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True
match_unstable_error = True
use_detached_boxes_dec_out = False

# ---- DN ----
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.5
dn_label_noise_ratio = 0.55
embed_init_tgt = True
dn_labelbook_size = 10

# ---- EMA ----
use_ema = True
ema_decay = 0.9997
ema_epoch = 0

# ---- 类别加权 ----
use_visdrone_class_weights = True
visdrone_class_weights = [
    1.0,   # pedestrian  (AP50=0.65)
    1.8,   # people      (AP50=0.53, Recall 低)
    4.0,   # bicycle     (AP50=0.39, 很低)
    0.6,   # car         (AP50=0.89, 已饱和，降权)
    2.5,   # van         (AP50=0.58)
    3.5,   # truck       (AP50=0.44, 低)
    3.0,   # tricycle    (AP50=0.43)
    7.0,   # awning-tri  (AP50=0.21, 最差)
    2.0,   # bus         (AP50=0.70)
    1.0,   # motor       (AP50=0.72)
]

# ---- 数据增强开关 ----
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

# ---- 杂项 ----
frozen_weights = None
checkpoint_every_iters = 0
cuda_empty_cache_every_iters = 200
visdrone_retrain_plan_aug = False

