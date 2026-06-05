# 毕业论文资料「DINO精度的提升」+ Deep Research 摘要中、当前仓库已可落地的部分：
# - 充分训练、SIoU、RandomSizeCrop 上界 800、编码器通道注意力：继承自 DINO_4scale_visdrone_thesis.py
# - 开启 EMA：验证集上常能稳定提升 mAP（与 DINO 官方实践一致）
#
# 已实现并可直接用（优先冲阶段A阈值，尤其 APs/Recall）：
# - Copy-Paste：thesis 已开
# - Mosaic（轻量 2x2）：thesis 已开（train_mosaic_prob / train_mosaic_size）
#
# 与目录 Verification/...ablation_standard_one 对比：该次为 12 epoch + GIoU + crop≤600 的消融；
# 要冲精度请用本配置或 thesis 完整栈长训，而非短消融 config。
_base_ = ['DINO_4scale_visdrone_thesis.py']

use_ema = True
ema_decay = 0.9997
ema_epoch = 0

# 冲阶段A阈值：适度提高 Copy-Paste 强度（更利于 AP_small/Recall）
copy_paste_prob = 0.65
copy_paste_max_objects = 5
