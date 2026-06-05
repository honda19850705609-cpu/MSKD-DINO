"""DINO-DETR: export to ONNX (Day 3).

Wraps the model so it takes a plain image tensor, exports with opset 17
(needed for grid_sample in the pure-PyTorch deformable attention), declares
dynamic spatial axes, and verifies the ONNX output against PyTorch.

Run from the project root:
    python export_onnx.py
"""

import torch
import torch.nn as nn
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig

CONFIG_PATH = "config/DINO/DINO_5scale_visdrone_lightweight_100ep.py"
CKPT_PATH = "/root/autodl-tmp/Day3/checkpoint_best_ema.pth"
ONNX_PATH = "/root/autodl-tmp/Day3/dino_visdrone.onnx"
OPSET = 17
DUMMY_H, DUMMY_W = 800, 1333


class DinoExportWrapper(nn.Module):
    """Wrap DINO so forward takes a plain (B,3,H,W) tensor and returns
    just the two tensors we care about: class logits and boxes."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        out = self.model(images)
        # DINO returns a dict; extract the final-layer predictions
        return out["pred_logits"], out["pred_boxes"]


def main():
    print(">>> building model ...")
    args = SLConfig.fromfile(CONFIG_PATH)
    args.device = "cpu"
    args.dataset_file = "coco"
    model, _, _ = build_model_main(args)

    print(">>> loading checkpoint ...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    wrapper = DinoExportWrapper(model).eval()

    dummy = torch.randn(1, 3, DUMMY_H, DUMMY_W)

    print(">>> sanity-check PyTorch forward through wrapper ...")
    with torch.no_grad():
        logits_pt, boxes_pt = wrapper(dummy)
    print(f"    pred_logits: {tuple(logits_pt.shape)}")
    print(f"    pred_boxes : {tuple(boxes_pt.shape)}")

    print(f">>> exporting to ONNX (opset {OPSET}) ...")
    torch.onnx.export(
        wrapper,
        dummy,
        ONNX_PATH,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        opset_version=OPSET,
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "pred_logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
        do_constant_folding=False,
    )
    print(f"    saved: {ONNX_PATH}")

    print(">>> verifying ONNX model structure ...")
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("    onnx.checker: OK")

    print(">>> comparing ONNXRuntime output vs PyTorch ...")
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"images": dummy.cpu().numpy()})
    logits_onnx, boxes_onnx = ort_out

    diff_logits = np.abs(logits_pt.cpu().numpy() - logits_onnx).max()
    diff_boxes = np.abs(boxes_pt.cpu().numpy() - boxes_onnx).max()
    print(f"    max|logits diff|: {diff_logits:.6f}")
    print(f"    max|boxes  diff|: {diff_boxes:.6f}")
    tol = 1e-3
    if diff_logits < tol and diff_boxes < tol:
        print(f"\n>>> ONNX export SUCCESS. Outputs match PyTorch within {tol}.")
    else:
        print(f"\n>>> WARNING: output diff exceeds {tol}. ONNX exported but verify numerics.")


if __name__ == "__main__":
    main()
