# DINO lightweight + precision boost (100ep)

data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 864, 928, 1000, 1100, 1200, 1400, 1500]
data_aug_max_size = 2000
data_aug_scales2_resize = [600, 800, 1000, 1200, 1500]
data_aug_scales2_crop = [600, 1200]
data_aug_scale_overlap = None

num_classes = 10
modelname = 'dino'
backbone = 'resnet50'
use_checkpoint = False
dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [0, 1, 2, 3]
num_feature_levels = 5
p2_proj_stride = 2
use_bifpn_neck = True
bifpn_repeats = 1
backbone_freeze_keywords = None

enc_layers = 4
dec_layers = 4
unic_layers = 0
pre_norm = False
dim_feedforward = 1024
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 1200
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
num_select = 400
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True

lr = 1e-4
param_dict_type = 'default'
lr_backbone = 1e-5
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
lr_encoder_mult = 1.3
ddetr_lr_param = False
weight_decay = 0.0001
clip_max_norm = 0.1

batch_size = 4
epochs = 100
lr_drop = 999
onecyclelr = False
multi_step_lr = True
lr_drop_list = [75, 92]
lr_warmup_epochs = 5
save_checkpoint_interval = 1

cls_loss_coef = 3.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
box_iou_loss_type = 'siou'
focal_alpha = 0.65
mask_loss_coef = 1.0
dice_loss_coef = 1.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False

set_cost_class = 3.0
set_cost_bbox = 5.0
set_cost_giou = 2.0

decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1
dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True
match_unstable_error = True
use_detached_boxes_dec_out = False

use_dn = True
dn_number = 150
dn_box_noise_scale = 0.5
dn_label_noise_ratio = 0.55
embed_init_tgt = True
dn_labelbook_size = 10

use_ema = True
ema_decay = 0.9997
ema_epoch = 0

use_visdrone_class_weights = True
visdrone_class_weights = [1.0, 1.8, 4.0, 0.6, 2.5, 3.5, 3.0, 7.0, 2.0, 1.0]

copy_paste_prob = 0.85
copy_paste_max_objects = 10
copy_paste_max_src_area_pixels = 2048
train_color_jitter = True
train_motion_blur_prob = 0.2
train_motion_blur_kernel = 11
train_class_balance_sampler = True
train_balance_small_area_max = 1024
train_balance_small_boost_scale = 1.0
train_gaussian_blur_prob = 0.08
train_gaussian_blur_radius = 0.8
train_mosaic_prob = 0.35
train_mosaic_size = (1280, 800)

frozen_weights = None
checkpoint_every_iters = 0
cuda_empty_cache_every_iters = 200
visdrone_retrain_plan_aug = False
laptop_stable_training = False
laptop_stable_num_workers_cap = 8
