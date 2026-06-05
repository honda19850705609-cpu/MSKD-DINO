"""DINO-DETR: export to ONNX (Day 3).

Wraps the model so it takes a plain image tensor, exports with opset 17
(needed for grid_sample in the pure-PyTorch deformable attention), declares
dynamic spatial axes, and verifies the ONNX output against PyTorch.

Run from the project root:
    python export_onnx.py --checkpoint path/to/checkpoint.pth --output dino_visdrone.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from main import build_model_main
from util.slconfig import SLConfig


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


def parse_args():
    parser = argparse.ArgumentParser("Export MSKD-DINO to ONNX")
    parser.add_argument(
        "--config",
        default="config/DINO/DINO_5scale_visdrone_lightweight_100ep.py",
        help="Model config used to build the checkpoint architecture.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth file.")
    parser.add_argument("--output", default="outputs/dino_visdrone.onnx", help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset. Opset 17 supports grid_sample.")
    parser.add_argument("--height", type=int, default=800, help="Dummy export image height.")
    parser.add_argument("--width", type=int, default=1333, help="Dummy export image width.")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Max absolute diff tolerance.")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    output_path = Path(cli_args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(">>> building model ...")
    args = SLConfig.fromfile(cli_args.config)
    args.device = "cpu"
    args.dataset_file = "coco"
    model, _, _ = build_model_main(args)

    print(">>> loading checkpoint ...")
    ckpt = torch.load(cli_args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    wrapper = DinoExportWrapper(model).eval()

    dummy = torch.randn(1, 3, cli_args.height, cli_args.width)

    print(">>> sanity-check PyTorch forward through wrapper ...")
    with torch.no_grad():
        logits_pt, boxes_pt = wrapper(dummy)
    print(f"    pred_logits: {tuple(logits_pt.shape)}")
    print(f"    pred_boxes : {tuple(boxes_pt.shape)}")

    print(f">>> exporting to ONNX (opset {cli_args.opset}) ...")
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        opset_version=cli_args.opset,
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "pred_logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
        do_constant_folding=False,
    )
    print(f"    saved: {output_path}")

    print(">>> verifying ONNX model structure ...")
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("    onnx.checker: OK")

    print(">>> comparing ONNXRuntime output vs PyTorch ...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"images": dummy.cpu().numpy()})
    logits_onnx, boxes_onnx = ort_out

    diff_logits = np.abs(logits_pt.cpu().numpy() - logits_onnx).max()
    diff_boxes = np.abs(boxes_pt.cpu().numpy() - boxes_onnx).max()
    print(f"    max|logits diff|: {diff_logits:.6f}")
    print(f"    max|boxes  diff|: {diff_boxes:.6f}")
    if diff_logits < cli_args.tolerance and diff_boxes < cli_args.tolerance:
        print(f"\n>>> ONNX export SUCCESS. Outputs match PyTorch within {cli_args.tolerance}.")
    else:
        print(f"\n>>> WARNING: output diff exceeds {cli_args.tolerance}. ONNX exported but verify numerics.")


if __name__ == "__main__":
    main()
