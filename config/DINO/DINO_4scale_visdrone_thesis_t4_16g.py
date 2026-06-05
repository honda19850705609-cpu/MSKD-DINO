# NVIDIA T4 / 真・小显存（约 15～16GB）专用：论文栈收紧尺度与 DN
# （勿与 Colab「G4」大显存实例混淆，后者见 DINO_4scale_visdrone_thesis_g4.py）
#
#   python main.py -c config/DINO/DINO_4scale_visdrone_thesis_t4_16g.py --amp ...
# 若仍 OOM: --options batch_size=1
# 冒烟: DINO_4scale_visdrone_thesis_t4_16g_quick.py
_base_ = ['DINO_4scale_visdrone_thesis.py']

batch_size = 2
lr = 5e-5
lr_backbone = 5e-6

data_aug_scales2_crop = [384, 640]
data_aug_scales = [480, 512, 544, 576, 608, 640]
dn_number = 64

cuda_empty_cache_every_iters = 100
checkpoint_every_iters = 500
