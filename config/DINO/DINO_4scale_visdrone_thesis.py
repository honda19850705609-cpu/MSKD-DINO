# VisDrone-DET 10 类 + 毕业论文（任务书）实验默认：与 tools/visdrone_cleaned_txt_to_coco.py 一致
# 要求 COCO JSON 中 category_id 为 0..9（与 num_classes=10 对齐）；否则启动时 util/coco_ann_checks 会报错。
#
# 对齐《DINO-DETR优化VisDrone数据集》、任务书、《DINO小目标检测优化》、recall_aps.pdf、
# 《提升DINO在VisDrone的检测性能》（编码端略高 LR、DN 噪声、弱高斯模糊；未引入 RoPE/P2/冻结骨干等大改）：
# - mAP@0.5 ≥ 35%、兼顾 FPS（推理请用 --amp + tools/benchmark.py；部署可再降分辨率/num_select）
# - SIoU 边界框损失、更长训练、更大随机裁剪（利于小目标）、编码器通道注意力（轻量特征增强）
# - 主干 use_checkpoint 仅对 Swin 有效；R50 请用 --amp 与较小 batch 控显存
# - 真 A100-80G 请用：config/DINO/DINO_4scale_visdrone_thesis_a100.py（MSDA 编译 sm_80）；交互训练预设① 第一项亦为 A100 快速档
# - Colab 大显存「G4」（~80G，多为 Blackwell 等）请用：…_thesis_g4.py（MSDA 勿强编 8.0）
# - T4 / 真 16GB：…_thesis_t4_16g.py
# - 大显存要快：上述配置均建议加 --amp
# - 快速验证管线（少 epoch、非正式指标）：config/DINO/DINO_4scale_visdrone_thesis_quick.py + --amp
# - 毕业论文 PDF「DINO精度的提升」：在论文栈上再加 EMA → DINO_4scale_visdrone_graduation_max_ap.py（及 *_a100 / *_g4 / *_t4_16g）
_base_ = ['DINO_4scale.py']

num_classes = 10
dn_labelbook_size = 10

# --- 长训（论文建议 100 epoch；任务书需充分收敛）---
epochs = 100
batch_size = 4
# 论文：Transformer 1e-4、主干约 0.1×；与 param_dict_type=default（backbone 用 lr_backbone）一致
lr = 1e-4
lr_backbone = 1e-5
# 检测性能 PDF：浅层/编码端更利于小目标特征；略提高 transformer.encoder 学习率（温和 1.3×）
lr_encoder_mult = 1.3
multi_step_lr = True
# 中后段两次降 lr（可按验证曲线微调）
lr_drop_list = [70, 90]

# 回归与匹配：SIoU；略加大 IoU 项与匹配代价，利于框质量与 Recall（recall_aps.pdf）
box_iou_loss_type = 'siou'
giou_loss_coef = 2.5
set_cost_giou = 2.5
# 后处理多保留高分 query，略抬召回（部署仍可降 num_select）
num_select = 350
# 分类：按 focal_alpha=0.25（DINO 默认）更稳；过小可能损 Recall/APs
focal_alpha = 0.25

# 高遮挡场景：略增强去噪查询噪声（仍兼容原 DN 训练）
dn_box_noise_scale = 0.42
dn_label_noise_ratio = 0.52

# 编码器层内轻量通道注意力（与论文 DAFM 思想接近但计算更轻，符合「简化改进」方向）
add_channel_attention = True

# 数据增强：略扩训练短边上限（recall_aps：多尺度 640–1024 方向），与小目标 Copy-Paste 叠加；T4 配置会收紧
data_aug_scales = [
    480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 1000, 1100, 1200,
]
data_aug_max_size = 1400
data_aug_scales2_crop = [384, 800]

# 笔记本易死机、整 epoch 跑不完时：每 N 个 train iteration 写 checkpoint_emergency.pth（续训见 main 日志说明）；0=关闭
checkpoint_every_iters = 0
laptop_stable_training = False
laptop_stable_num_workers_cap = 8
cuda_empty_cache_every_iters = 200

# Deep Research +《DINO小目标检测优化》：warmup、小目标 Copy-Paste、光度、运动模糊、类均衡采样
lr_warmup_epochs = 5
copy_paste_prob = 0.5
copy_paste_max_objects = 3
# COCO 小目标定义：面积 < 32×32；优先只裁贴小实例，若无则退回任意框
copy_paste_max_src_area_pixels = 1024
train_color_jitter = True
train_motion_blur_prob = 0.2
train_motion_blur_kernel = 11
train_gaussian_blur_prob = 0.08
train_gaussian_blur_radius = 0.8
train_class_balance_sampler = True
# 在类均衡采样上再提高「小目标占比高」图像的抽中概率，与 copy_paste_max_src_area_pixels=1024 一致
train_balance_small_area_max = 1024
train_balance_small_boost_scale = 0.45

# Mosaic：增加小目标出现率与背景多样性（先轻量固定 2x2）
train_mosaic_prob = 0.30
train_mosaic_size = (1280, 800)
