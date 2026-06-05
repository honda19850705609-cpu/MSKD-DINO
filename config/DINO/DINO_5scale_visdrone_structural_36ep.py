# VisDrone 结构性升级（P2 + 5-level + BiFPN + 类加权 focal）— 36 epoch 快速验证档
# 与文档对齐：epochs=36，第 30 轮起 MultiStepLR 乘 gamma（默认 0.1），lr=2e-4，lr_backbone=2e-5
#
# 用法：
#   python main.py -c config/DINO/DINO_5scale_visdrone_structural_36ep.py --amp ...
#
# 说明：比 50ep 冲线版更早结束；若指标未达标可改回 DINO_5scale_visdrone_structural.py 或 *_stageA_target_50ep 系。
_base_ = ['DINO_5scale_visdrone_structural.py']

epochs = 36
multi_step_lr = True
# PyTorch MultiStepLR：完成 epoch 29 后 last_epoch→30，触发衰减；epoch 30–35 用低学习率
lr_drop_list = [30]

lr = 2e-4
lr_backbone = 2e-5
