# VisDrone 结构升级 + 高分辨率训练增强（对应 structural_upgrade.md 的“大尺度 + 大 crop”）
#
# 用法：
#   python main.py -c config/DINO/DINO_5scale_visdrone_structural_hires.py --amp ...
_base_ = ['DINO_5scale_visdrone_structural.py']

# --------- Hires augmentation (train) ---------
# 正常分支（最终训练尺寸范围）
data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 1333

# 大尺度分支：直接 RandomResize 到更大短边，并允许更大 max_size
data_aug_scales_large = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
data_aug_max_size_large = 2000

# crop 分支：先放大再 crop（核心：放大小目标有效尺度）
data_aug_scales2_resize = [800, 1000, 1200, 1500, 1800]
data_aug_scales2_crop = [800, 1200]

# --------- Hires augmentation (val) ---------
# 验证也用更大分辨率（更贴合小目标）
data_aug_val_scales = [1000]
data_aug_val_max_size = 1667

# Hires 下显存更吃紧：保持更保守的 batch；如有余量再调高
batch_size = 2

