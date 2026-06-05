# 真 NVIDIA A100（40GB / 80GB）上跑 VisDrone 论文配置（MSDA 请按 sm_80 编译，勿与 Colab G4 混用）
#
# 使用（务必开混精，显存与速度都更好）:
#   python main.py -c config/DINO/DINO_4scale_visdrone_thesis_a100.py --amp ...
# 数据线程（若 CPU 够）可加大加载并行:
#   --num_workers 12
#
# 80GB 且仍有余量时可试: --options batch_size=12 lr=2.5e-4 lr_backbone=2.5e-5
# 40GB 若 OOM: --options batch_size=6 lr=1.5e-4 lr_backbone=1.5e-5
_base_ = ['DINO_4scale_visdrone_thesis.py']

# batch 4→8：近似线性放大 lr（相对默认 thesis）
batch_size = 8
lr = 2e-4
lr_backbone = 2e-5

# 关掉周期性 empty_cache，减少同步、提升迭代速度（A100 显存充足）
cuda_empty_cache_every_iters = 0

# 关闭按 iteration 写紧急 checkpoint，减少磁盘阻塞（需要时可改回 500）
checkpoint_every_iters = 0

# 云端大实例一般不需要笔记本稳态限制
laptop_stable_training = False
