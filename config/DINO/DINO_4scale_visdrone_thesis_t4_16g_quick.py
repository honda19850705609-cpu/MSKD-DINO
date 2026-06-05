# T4 / ~16GB 上的 quick 冒烟（继承 quick，batch=2）
_base_ = ['DINO_4scale_visdrone_thesis_quick.py']

batch_size = 2
lr = 2.5e-5
lr_backbone = 2.5e-6

cuda_empty_cache_every_iters = 100
checkpoint_every_iters = 500
