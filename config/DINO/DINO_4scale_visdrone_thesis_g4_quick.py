# Colab 大显存 G4（~80GB）quick；A100-80G 请用 DINO_4scale_visdrone_thesis_a100_quick.py
_base_ = ['DINO_4scale_visdrone_thesis_quick.py']

batch_size = 10
lr = 1.25e-4
lr_backbone = 1.25e-5

cuda_empty_cache_every_iters = 0
checkpoint_every_iters = 0
