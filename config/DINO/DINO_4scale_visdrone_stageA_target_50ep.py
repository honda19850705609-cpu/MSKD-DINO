# 阶段A：优先达标（mAP@0.5 ≥ 0.55, APs ≥ 0.25, Recall ≥ 0.60, F1 ≥ 0.65）
# 面向 “50 epoch 左右” 的冲线配置：论文栈 + EMA + Copy-Paste + Mosaic + 类均衡采样
#
# 用法：
#   python main.py -c config/DINO/DINO_4scale_visdrone_stageA_target_50ep.py --amp ...
_base_ = ['DINO_4scale_visdrone_graduation_max_ap.py']

# 目标轮次
epochs = 50

# 50ep 下把降 LR 提前到中后段（比 thesis 的 70/90 更适合 50ep 资源）
multi_step_lr = True
lr_drop_list = [35, 45]

# F1/Precision 需要一定克制：不要把 num_select 拉太大（过多低分框会拉低 Precision）
num_select = 300

# 进一步偏向小目标：略增强小目标图权重与 Copy-Paste
train_balance_small_boost_scale = 0.55
copy_paste_prob = 0.70
copy_paste_max_objects = 5

# Mosaic 保持轻量，不要过高概率（过强会影响收敛稳定）
train_mosaic_prob = 0.30
train_mosaic_size = (1280, 800)

