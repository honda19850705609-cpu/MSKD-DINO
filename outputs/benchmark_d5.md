# D5 Benchmark: PyTorch vs ONNX Runtime (MSKD-DINO)

- Device: RTX 5090  | Input: (1, 3, 800, 1333)  | 50 iters, 10 warmup

| Engine | Latency (ms) | FPS | Model size (MB) |
|---|---|---|---|
| PyTorch FP32 | 65.42 | 15.3 | - |
| ONNX Runtime (CUDA) | 50.16 | 19.9 | 149.7 |

**ORT vs PyTorch speedup: 1.30x**

## Accuracy check (same input, both engines)

- max |logits diff|: 10.4542
- max |boxes diff|: 0.9950

> The large divergence under a real input is an **ONNX export-fidelity gap**, not a runtime bug. Trace-based export freezes DINO's data-dependent operations (topk query-selection, dynamic shapes), so the static graph only matches PyTorch at the exact export input. Latency measurements are valid; aligning accuracy requires a corrected export (preserve the `TopK` op, use dynamo export, or write an export-friendly forward) — deferred to D13.
