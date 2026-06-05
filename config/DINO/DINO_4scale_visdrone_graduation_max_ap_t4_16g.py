# T4 / 16GB 档 + 毕业论文 PDF 建议的 EMA（小显存仍用论文栈收紧项，见 thesis_t4_16g）
_base_ = ['DINO_4scale_visdrone_thesis_t4_16g.py']

use_ema = True
ema_decay = 0.9997
ema_epoch = 0
