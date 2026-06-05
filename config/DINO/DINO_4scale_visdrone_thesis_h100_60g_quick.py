# NVIDIA H100 60GB：快速冒烟/短测（MSDA 请用 sm_90 编译）
#
#   python main.py -c config/DINO/DINO_4scale_visdrone_thesis_h100_60g_quick.py --amp ...
_base_ = ['DINO_4scale_visdrone_thesis_quick.py']

# quick 栈本身更轻；60G 可把 batch 稍提，提高吞吐
batch_size = 8
lr = 1e-4
lr_backbone = 1e-5

cuda_empty_cache_every_iters = 0
checkpoint_every_iters = 0
laptop_stable_training = False

