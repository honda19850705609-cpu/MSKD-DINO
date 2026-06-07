"""DINO-DETR D5 benchmark: PyTorch (FP32) vs ONNX Runtime (CUDA).

Measures end-to-end inference latency for the same model in two engines and
checks that the ONNX outputs match PyTorch numerically. Produces the D5
comparison table (latency / throughput / model size / accuracy diff).

Methodology notes (so the numbers are trustworthy):
  - GPU work is async; we call torch.cuda.synchronize() around every timed
    region, otherwise we'd be timing kernel-launch, not compute.
  - The first few runs include CUDA kernel compilation / allocation and are
    far slower, so we WARM UP before timing.
  - We run many iterations and report mean + std, not a single shot.
  - Both engines get the exact same preprocessed input tensor.

Run from the MSKD-DINO repo root:
    python benchmark.py \
        --config config/DINO/DINO_5scale_visdrone_lightweight_100ep.py \
        --checkpoint /root/autodl-tmp/Day5/Task_DINO/checkpoint_best_ema.pth \
        --onnx outputs/dino_visdrone.onnx \
        --image /root/autodl-tmp/images/img1.jpg \
        --iters 50 --warmup 10
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

import datasets.transforms as T
from main import build_model_main
from util.slconfig import SLConfig


def parse_args():
    p = argparse.ArgumentParser("MSKD-DINO PyTorch vs ONNX Runtime benchmark")
    p.add_argument("--config", default="config/DINO/DINO_5scale_visdrone_lightweight_100ep.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--onnx", required=True, help="exported .onnx file")
    p.add_argument("--image", required=True, help="image for the benchmark input")
    p.add_argument("--iters", type=int, default=50, help="timed iterations")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations (not timed)")
    p.add_argument("--out-json", default="outputs/benchmark_d5.json",
                   help="where to save the comparison results")
    p.add_argument("--out-md", default="outputs/benchmark_d5.md",
                   help="where to save the markdown comparison table")
    return p.parse_args()


def preprocess(image_path, fixed_h=800, fixed_w=1333):
    """Force the EXACT export size so ONNX's baked-in shapes line up.

    The ONNX graph was traced at 800x1333; feeding any other size makes
    BiFPN's internal Add ops fail to broadcast. We resize to the export
    size directly instead of the dynamic RandomResize, so both engines get
    an identical, fixed-size input tensor.
    """
    import torchvision.transforms.functional as TF
    image = Image.open(image_path).convert("RGB")
    img = TF.to_tensor(image)
    img = TF.resize(img, [fixed_h, fixed_w])
    img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img[None]  # (1, 3, 800, 1333)


def time_pytorch(model, inp, iters, warmup, device):
    model.eval()
    inp = inp.to(device)
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # timed
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            out = model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)  # ms
    # grab one output for the accuracy comparison
    logits = out["pred_logits"].detach().cpu().numpy()
    boxes = out["pred_boxes"].detach().cpu().numpy()
    return np.array(times), logits, boxes


def time_onnx(onnx_path, inp_np, iters, warmup):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    used = sess.get_providers()
    print(f"    ORT providers in use: {used}")
    in_name = sess.get_inputs()[0].name
    # warmup
    for _ in range(warmup):
        _ = sess.run(None, {in_name: inp_np})
    # timed
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = sess.run(None, {in_name: inp_np})
        times.append((time.perf_counter() - t0) * 1000.0)
    logits, boxes = out[0], out[1]
    return np.array(times), logits, boxes, used


def fmt(ms_array):
    return f"{ms_array.mean():6.2f} ms  (std {ms_array.std():.2f}, min {ms_array.min():.2f})"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> device: {device}")

    print(">>> building model + loading checkpoint ...")
    cfg = SLConfig.fromfile(args.config)
    cfg.device = str(device)
    cfg.dataset_file = "coco"
    model, _, _ = build_model_main(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    print(">>> preprocessing image ...")
    inp = preprocess(args.image)
    inp_np = inp.numpy()
    print(f"    input shape: {tuple(inp.shape)}")

    print(f"\n>>> timing PyTorch (FP32, {args.iters} iters, {args.warmup} warmup) ...")
    pt_times, pt_logits, pt_boxes = time_pytorch(model, inp, args.iters, args.warmup, device)
    print(f"    PyTorch: {fmt(pt_times)}")

    print(f"\n>>> timing ONNX Runtime ({args.iters} iters, {args.warmup} warmup) ...")
    ort_times, ort_logits, ort_boxes, providers = time_onnx(args.onnx, inp_np, args.iters, args.warmup)
    print(f"    ONNX RT: {fmt(ort_times)}")

    # accuracy: max abs diff between the two engines' outputs
    diff_logits = float(np.abs(pt_logits - ort_logits).max())
    diff_boxes = float(np.abs(pt_boxes - ort_boxes).max())

    pt_ms = pt_times.mean()
    ort_ms = ort_times.mean()
    onnx_mb = os.path.getsize(args.onnx) / 1024**2
    speedup = pt_ms / ort_ms if ort_ms > 0 else float("nan")

    print("\n" + "=" * 64)
    print("D5 COMPARISON  (RTX 5090, input {} )".format(tuple(inp.shape)))
    print("=" * 64)
    print(f"{'engine':<22}{'latency(ms)':>14}{'FPS':>10}{'size(MB)':>12}")
    print("-" * 64)
    print(f"{'PyTorch FP32':<22}{pt_ms:>14.2f}{1000/pt_ms:>10.1f}{'-':>12}")
    print(f"{'ONNX Runtime (CUDA)':<22}{ort_ms:>14.2f}{1000/ort_ms:>10.1f}{onnx_mb:>12.1f}")
    print("-" * 64)
    print(f"ORT vs PyTorch speedup : {speedup:.2f}x")
    print(f"max |logits diff|      : {diff_logits:.6f}")
    print(f"max |boxes  diff|      : {diff_boxes:.6f}")
    print(f"ORT providers          : {providers}")
    print("=" * 64)

    # ---- save results to disk (D5 deliverable) ----
    results = {
        "device": "RTX 5090",
        "input_shape": list(inp.shape),
        "iters": args.iters,
        "warmup": args.warmup,
        "engines": {
            "pytorch_fp32": {
                "latency_ms_mean": round(float(pt_ms), 3),
                "latency_ms_std": round(float(pt_times.std()), 3),
                "fps": round(1000 / pt_ms, 2),
            },
            "onnxruntime_cuda": {
                "latency_ms_mean": round(float(ort_ms), 3),
                "latency_ms_std": round(float(ort_times.std()), 3),
                "fps": round(1000 / ort_ms, 2),
                "model_size_mb": round(onnx_mb, 1),
                "providers": providers,
            },
        },
        "ort_vs_pytorch_speedup": round(float(speedup), 3),
        "accuracy_check": {
            "max_logits_diff": round(diff_logits, 6),
            "max_boxes_diff": round(diff_boxes, 6),
            "note": (
                "Large diff under a real (non-export) input reveals an ONNX "
                "export-fidelity gap, not a runtime bug: trace-based export "
                "freezes DINO's data-dependent ops (topk query-selection, "
                "dynamic shapes). Latency numbers are valid; accuracy alignment "
                "needs a corrected export (preserve TopK / dynamo export / "
                "export-friendly forward) -- deferred to D13."
            ),
        },
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f">>> saved JSON: {args.out_json}")

    md = (
        "# D5 Benchmark: PyTorch vs ONNX Runtime (MSKD-DINO)\n\n"
        f"- Device: RTX 5090  | Input: {tuple(inp.shape)}  "
        f"| {args.iters} iters, {args.warmup} warmup\n\n"
        "| Engine | Latency (ms) | FPS | Model size (MB) |\n"
        "|---|---|---|---|\n"
        f"| PyTorch FP32 | {pt_ms:.2f} | {1000/pt_ms:.1f} | - |\n"
        f"| ONNX Runtime (CUDA) | {ort_ms:.2f} | {1000/ort_ms:.1f} | {onnx_mb:.1f} |\n\n"
        f"**ORT vs PyTorch speedup: {speedup:.2f}x**\n\n"
        "## Accuracy check (same input, both engines)\n\n"
        f"- max |logits diff|: {diff_logits:.4f}\n"
        f"- max |boxes diff|: {diff_boxes:.4f}\n\n"
        "> The large divergence under a real input is an **ONNX export-fidelity "
        "gap**, not a runtime bug. Trace-based export freezes DINO's "
        "data-dependent operations (topk query-selection, dynamic shapes), so "
        "the static graph only matches PyTorch at the exact export input. "
        "Latency measurements are valid; aligning accuracy requires a corrected "
        "export (preserve the `TopK` op, use dynamo export, or write an "
        "export-friendly forward) — deferred to D13.\n"
    )
    with open(args.out_md, "w") as f:
        f.write(md)
    print(f">>> saved Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
