# MSKD-DINO — Deployment & Optimization for VisDrone Detection

Deployment-side work for a DINO-DETR detector trained on VisDrone (dense small-object
aerial detection). This repository contains the scripts and notes for taking a trained
DINO-DETR checkpoint through an end-to-end deployment pipeline:

```
PyTorch  ->  ONNX  ->  TensorRT
```

The detector itself is a 5-scale DINO with a ResNet-50 backbone and several
modifications (BiFPN neck, P2 high-resolution level, channel attention, SIoU loss,
VisDrone class weighting), trained for 10 VisDrone classes.

> This repo holds the **deployment scripts and documentation**, not the full DINO
> training codebase or model weights. The base detector builds on the IDEA-Research
> DINO codebase.

## Contents

| File | Purpose |
|------|---------|
| `verify_inference.py` | Build the model from its training config, load the checkpoint, run one PyTorch inference and print detections. |
| `export_onnx.py` | Export the model to ONNX (opset 17), with dynamic spatial axes, and validate the ONNX output against PyTorch. |
| `setup_dino.sh` | Reproducible environment setup (CUDA paths, dependencies, deformable-attention build). |
| `logs/` | Daily engineering logs documenting the process, problems, and fixes. |

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

## Status

- [x] PyTorch inference verified (219 detections on a VisDrone aerial image)
- [x] ONNX export validated (onnx.checker OK, ONNXRuntime matches PyTorch)
- [ ] TensorRT engine + before/after comparison table (AP/F1, latency, size, VRAM)

## Usage

Run from the DINO project root (where `main.py` lives), after placing these scripts there:

```bash
# 1. environment
bash setup_dino.sh

# 2. verify the model loads and runs
python verify_inference.py

# 3. export and validate ONNX
python export_onnx.py
```

Edit the paths at the top of each script (`CONFIG_PATH`, `CKPT_PATH`, `IMAGE_PATH`,
`ONNX_PATH`) to match your environment.
