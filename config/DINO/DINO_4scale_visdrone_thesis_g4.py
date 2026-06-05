# Colab Pro+「G4」等大显存实例（约 80GB 级；常为 Blackwell 等新架构，非 A100）
# 训练超参与 A100-80G 同档；MSDA 编译请勿用 TORCH_CUDA_ARCH_LIST=8.0（见 util/msda_reinstall）。
# 真 A100-80G 请用：DINO_4scale_visdrone_thesis_a100.py
# T4 16GB 请用：DINO_4scale_visdrone_thesis_t4_16g.py
#
#   python main.py -c config/DINO/DINO_4scale_visdrone_thesis_g4.py --amp ...
# 有余量可试: --options batch_size=10 lr=2.5e-4 lr_backbone=2.5e-5
_base_ = ['DINO_4scale_visdrone_thesis_a100.py']
