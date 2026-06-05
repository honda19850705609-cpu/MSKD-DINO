# 快速冒烟 / 验证管线（不用于正式刷 mAP）
# - 默认 6 个 epoch（比极短冒烟略长，仍远少于正式 100）；增强上限 640；关通道注意力；GIoU；DN 减半
# - 正式实验请用 thesis.py；大显存 G4（~80G）用 thesis_g4；T4 用 thesis_t4_16g
#
# 建议（尽量快）:
#   python main.py -c config/DINO/DINO_4scale_visdrone_thesis_quick.py --amp --output_dir ... [数据参数]
# 更短: --options epochs=2 ；更长: --options epochs=10
# 再省时间可减 DataLoader 进程: --num_workers 4
_base_ = ['DINO_4scale_visdrone_thesis.py']

epochs = 6
batch_size = 8

# 单步更快：最大尺度 640、裁剪上界 640（相对 thesis 800）
data_aug_scales = [480, 512, 544, 576, 608, 640]
data_aug_scales2_crop = [384, 640]

# encoder 通道注意力 off：结构更简单、省算力（与正式 thesis 不一致，仅快验）
add_channel_attention = False

# 匹配与回归用 GIoU，省 SIoU 矩阵开销
box_iou_loss_type = 'giou'

# 去噪查询减半，减轻每 iteration 负担
dn_number = 50

lr = 1e-4
lr_backbone = 1e-5

cuda_empty_cache_every_iters = 0
checkpoint_every_iters = 0

# 快验关闭 Copy-Paste / 长 warmup / 光度 / 运动模糊 / 类均衡，缩短单 epoch 时间
copy_paste_prob = 0.0
lr_warmup_epochs = 0
train_color_jitter = False
copy_paste_max_src_area_pixels = None
train_motion_blur_prob = 0.0
train_class_balance_sampler = False
train_balance_small_area_max = None
train_balance_small_boost_scale = 0.0
# 与正式 recall/小目标栈解耦，快验用默认匹配与 focal
giou_loss_coef = 2.0
set_cost_giou = 2.0
num_select = 300
focal_alpha = 0.25
lr_encoder_mult = 1.0
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
train_gaussian_blur_prob = 0.0
