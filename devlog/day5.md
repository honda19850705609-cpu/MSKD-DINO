# Day 5 — DINO-DETR: ONNX export + PyTorch vs ONNX Runtime benchmark

## What I did
Built the deployment-comparison link for the DINO direction. On a fresh AutoDL
RTX 5090 instance I: recompiled the deformable-attention CUDA op, re-exported
the trained checkpoint to ONNX (opset 17), installed onnxruntime-gpu, and wrote
a benchmark that times PyTorch (FP32) vs ONNX Runtime (CUDA) on the same input
and checks output agreement. TensorRT was the original D5 target but is not
installable here yet (see below), so ORT stands in as the acceleration baseline.

## Key data (RTX 5090, input 1x3x800x1333, 50 iters / 10 warmup)
| Engine | Latency (ms) | FPS | Size (MB) |
|---|---|---|---|
| PyTorch FP32 | 65.35 | 15.3 | - |
| ONNX Runtime (CUDA) | 49.84 | 20.1 | 149.7 |

ORT vs PyTorch speedup: **1.31x**. Benchmark methodology: warmup before timing,
`torch.cuda.synchronize()` around every timed region, mean+std over many iters,
identical fixed-size input to both engines.

## Bugs / blockers hit
1. **CUDA 13 + Blackwell too new for TensorRT.** `nvidia-smi` reports CUDA 13.0;
   no compatible stable TensorRT build installs cleanly. Per the "don't dig in,
   get the link working" rule, I substituted onnxruntime-gpu for the acceleration
   side and deferred TensorRT to a controlled environment (post-onboarding).
2. **Deformable-attention op `libc10.so` not found at import.** The compiled .so
   couldn't locate PyTorch's lib dir at runtime. Fixed by adding torch's lib path
   to `LD_LIBRARY_PATH` (persisted in `.bashrc`).
3. **ONNX BiFPN `Add` broadcast failure (24 vs 23 on dim 2).** The graph was
   traced at 800x1333 but the dynamic preprocessing produced 750x1333. Fixed by
   forcing the benchmark input to the exact export size.

## The real finding: ONNX export-fidelity gap
The accuracy check showed a large divergence under a real image:
`max|logits diff| = 10.45`, `max|boxes diff| = 0.99` — versus `< 1e-3` at
export time. This is not a runtime bug; it is fundamental to how trace-based
ONNX export handles DINO:

- A model is **weights (data) + computation structure**. ONNX export does NOT
  change weights; it freezes the *structure* into a static graph.
- Most ops (conv, matmul, relu, softmax) are **structurally fixed** — what they
  compute never depends on the input *values*, only the fixed structure. These
  export losslessly (CNNs are almost entirely this class).
- A few ops are **data-dependent decisions**: `topk` (DINO's query selection),
  `if tensor > x`, boolean indexing, dynamic shapes, `.item()`. Trace-based
  export records only the path the dummy input took and *freezes the decision*.
- DINO's accuracy/elegance comes precisely from such dynamic mechanisms, so
  naive trace export collapses a piecewise function into a single branch:
  it matches PyTorch only at the export input, and diverges elsewhere.

**Crucially this is an export-method limitation, not a "DINO can't be exported"
conclusion.** Correct paths exist: keep the `TopK` op live in the graph instead
of freezing its indices, use PyTorch 2.x dynamo export (preserves control flow)
instead of trace, or write an export-friendly forward (fixed query count, shapes
made explicit). DETR-family deployment is "hard but standard," and this gap is
exactly what TensorRT acceleration must clear *before* it is meaningful.

## Next (D13 / deepening)
Re-export with a fidelity-preserving method, drive the diff back to ~1e-3, then
attempt TensorRT in a CUDA-controlled environment for the real FP16 speedup.
