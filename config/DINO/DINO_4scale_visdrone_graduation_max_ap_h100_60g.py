# NVIDIA H100 60GB：毕业论文/PDF（论文栈 + EMA）速度档
#
#   python main.py -c config/DINO/DINO_4scale_visdrone_graduation_max_ap_h100_60g.py --amp ...
_base_ = ['DINO_4scale_visdrone_graduation_max_ap.py']

batch_size = 6
lr = 1.5e-4
lr_backbone = 1.5e-5

cuda_empty_cache_every_iters = 0
checkpoint_every_iters = 0
laptop_stable_training = False

