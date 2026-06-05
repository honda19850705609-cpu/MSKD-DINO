# MSKD-DINO — Accuracy-then-Compression for Dense Small-Object Aerial Detection

A full research pipeline built on **DINO-DETR** for dense, small-object detection on
**VisDrone** (10 classes, ~6,500 aerial images). The project follows a single
end-to-end thesis: **first push accuracy, then distill and compress, then deploy** —

```
Accuracy boost  ->  Knowledge distillation  ->  Lightweighting  ->  ONNX / FP16 deployment
```

The goal is to test whether an *accuracy-first, compression-second* route is effective on
a Transformer detector, rather than compressing a weak baseline from the start.

> **Attribution.** This work builds on the [IDEA-Research/DINO](https://github.com/IDEA-Research/DINO)
> codebase (Apache-2.0). The base detector, training engine, and deformable-attention op
> originate there. All VisDrone-specific architecture changes, the distillation pipeline,
> the lightweight student configs, and the deployment scripts in this repo are my own
> additions on top of that base. See `LICENSE` and the original repository for the base license.

---

## What I changed

**Accuracy (architecture + loss + training).** Added a **P2 high-resolution feature level**
and a **5-scale** input, a **BiFPN** bidirectional feature pyramid, and **channel attention**
in the encoder. On the loss side, replaced GIoU with **SIoU** and applied a
**class-weighted Focal Loss** for VisDrone's long-tail distribution. Training used
**Mosaic + directed Copy-Paste + imaging-degradation** augmentation, a
`WeightedRandomSampler` for long-tail classes, and **EMA** (α=0.9997) for stability.

**Knowledge distillation & lightweighting.** Used the high-accuracy model as a teacher and
trained a student via **logits KL distillation**, while compressing the Transformer
encoder/decoder depth, FFN width, and BiFPN depth.

**Deployment.** Verified real-device inference and exported to ONNX; see the engineering
notes below for the non-obvious problems encountered on Blackwell-generation GPUs.

---

## Key results

| Stage | Metric | Result |
|-------|--------|--------|
| Accuracy boost (vs. baseline) | mAP / AP50 / APs | **+0.046 / +0.073 / +0.065** |
| Accuracy boost | core-class F1 | **0.640 → 0.712** |
| Student (after distill + compress) | params / GFLOPs | **−21.5% / −43%** |
| Student | mAP / AP50 / APs | **0.320 / 0.557 / 0.245** (clearly above baseline and compression-only ablation) |
| Deployment (FP16, 1333×800) | FPS | **33.4 → 45.8 (+37.2%)** |
| Deployment | end-to-end speedup vs. FP32 | **1.82×** (1333×800), **1.92×** (2000×1200) |

The student keeps most of the teacher's accuracy at a fraction of the cost, supporting the
accuracy-first-then-compress hypothesis. See `figs/` for training curves and the comparison table.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `main.py`, `engine.py` | Training / evaluation entry point and loop. |
| `config/DINO/` | DINO configs, including the VisDrone thesis / lightweight / structural variants. |
| `models/dino/` | Detector, with my additions: `bifpn_neck.py`, channel attention, deformable transformer. |
| `models/dino/ops/` | Multi-scale deformable-attention op (CUDA source + pure-PyTorch fallback). |
| `datasets/` | Data pipeline, incl. `copy_paste.py`, `coco_balance.py`, VisDrone transforms. |
| `tools/` | Training menu, metric plotting, VisDrone→COCO conversion, prediction visualization. |
| `util/` | Metrics (`thesis_metrics.py`, `prf1_metrics.py`), GPU-adaptive helpers, logging. |
| `verify_inference.py` | Build model from config, load checkpoint, run one PyTorch inference. |
| `export_onnx.py` | Export to ONNX (opset 17, dynamic spatial axes) and validate vs. PyTorch. |
| `figs/` | Framework diagram, training curves, comparison table. |
| `logs/` | Daily engineering logs (process, problems, fixes). |

> Model weights (`*.pth`), the compiled deformable-attention `.so`, and the VisDrone dataset
> are **not** tracked here (see `.gitignore`). The `.so` is platform/Python-version specific and
> must be rebuilt locally.

---

## Key engineering notes

- **Deformable-attention ABI break.** On torch 2.8 + RTX 5090 (Blackwell), the custom
  `MultiScaleDeformableAttention` CUDA op fails at runtime with an undefined C10 symbol
  (signature changed in torch 2.8), and downgrading torch conflicts with the GPU's
  required CUDA 12.8 toolkit. Resolved by bypassing the compiled op and forcing the
  pure-PyTorch `grid_sample`-based path — which is also what ONNX export requires.
- **ONNX export of a large DETR graph.** The 1200-query / 5-scale graph segfaults during
  ONNX writing under GPU memory pressure with constant folding on. Exporting on CPU with
  `do_constant_folding=False` is a reliable fallback. Output validated via `onnx.checker`
  and ONNXRuntime (shapes and logit ranges match PyTorch).

---

## Reproducing

```bash
# 1. environment
# Follow INSTALL_CONDA.txt to install PyTorch, dependencies, and the
# MultiScaleDeformableAttention op for your CUDA / Visual Studio setup.

# 2. train on VisDrone converted to COCO format
bash scripts/DINO_train.sh /path/to/visdrone_coco

# 3. evaluate
bash scripts/DINO_eval.sh /path/to/visdrone_coco /path/to/checkpoint.pth

# 4. deployment: verify + export ONNX
python verify_inference.py \
  --checkpoint /path/to/checkpoint.pth \
  --image /path/to/test.jpg

python export_onnx.py \
  --checkpoint /path/to/checkpoint.pth \
  --output outputs/dino_visdrone.onnx
```

The train / eval scripts default to `config/DINO/DINO_5scale_visdrone_final.py`.
Pass a second argument to `scripts/DINO_train.sh` or a third argument to
`scripts/DINO_eval.sh` if you want to run another config. Model weights and the
VisDrone dataset are intentionally not tracked, so the reported numbers should be
treated as experiment results from my local runs unless you retrain with the same setup.

## Status

- [x] Accuracy-boost training pipeline (P2 / 5-scale / BiFPN / SIoU / Focal / aug)
- [x] Knowledge distillation + lightweight student
- [x] PyTorch inference verified; ONNX export validated (onnx.checker OK, ORT matches PyTorch)
- [ ] TensorRT engine + before/after table (AP/F1, latency, size, VRAM)

## License

Code in this repository follows the license of the upstream
[IDEA-Research/DINO](https://github.com/IDEA-Research/DINO) project (Apache-2.0); see `LICENSE`.
