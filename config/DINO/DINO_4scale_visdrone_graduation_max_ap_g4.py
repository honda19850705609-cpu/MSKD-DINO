# Colab G4 等大显存档 + 毕业论文 PDF 建议的 EMA
_base_ = ['DINO_4scale_visdrone_thesis_g4.py']

use_ema = True
ema_decay = 0.9997
ema_epoch = 0
