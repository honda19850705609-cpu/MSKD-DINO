# NVIDIA H100 60GB：偏速度的论文栈（MSDA 请用 sm_90 编译）
#
# 使用（建议混精，H100 更快）:
#   python main.py -c config/DINO/DINO_4scale_visdrone_thesis_h100_60g.py --amp ...
# 建议数据线程（CPU 够的话更快）:
#   --num_workers 16
#
# 若仍 OOM（极端大分辨率 batch 峰值）:
#   --options batch_size=5 lr=1.25e-4 lr_backbone=1.25e-5
_base_ = ['DINO_4scale_visdrone_thesis.py']

# thesis 默认 batch=4；H100 60G 通常可到 6（速度/吞吐更优，且相对稳）
batch_size = 6
lr = 1.5e-4
lr_backbone = 1.5e-5

# 显存充足时关闭周期性 empty_cache，减少同步
cuda_empty_cache_every_iters = 0

# 关闭按 iteration 写紧急 checkpoint，避免磁盘 IO 影响吞吐（需要时可改回 500）
checkpoint_every_iters = 0

# 云端大卡通常不需要笔记本稳态限制
laptop_stable_training = False

