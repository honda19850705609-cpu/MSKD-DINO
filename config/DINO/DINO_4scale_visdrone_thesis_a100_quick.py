# A100-80G 档上的 quick（与交互菜单「~80GB A100-80G」一致；非 Colab G4）
_base_ = ['DINO_4scale_visdrone_thesis_quick.py']

batch_size = 10
lr = 1.25e-4
lr_backbone = 1.25e-5

cuda_empty_cache_every_iters = 0
checkpoint_every_iters = 0
