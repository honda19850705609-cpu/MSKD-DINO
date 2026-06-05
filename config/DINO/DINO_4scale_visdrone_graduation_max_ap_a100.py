# A100 大 batch 档 + 毕业论文 PDF 建议的 EMA（同 graduation_max_ap 思想）
_base_ = ['DINO_4scale_visdrone_thesis_a100.py']

use_ema = True
ema_decay = 0.9997
ema_epoch = 0
