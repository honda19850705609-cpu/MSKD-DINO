"""DINO-DETR: load checkpoint and run one PyTorch inference.

Verifies the model can be built with the training config, load the VisDrone
checkpoint, and run a forward pass before attempting ONNX export.

Run from the project root:
    python verify_inference.py --checkpoint path/to/checkpoint.pth --image path/to/test.jpg
"""

import argparse

import torch
from PIL import Image

import datasets.transforms as T
from main import build_model_main
from util.slconfig import SLConfig

# VisDrone 10 classes (index 0..9)
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def parse_args():
    parser = argparse.ArgumentParser("Verify one MSKD-DINO PyTorch inference")
    parser.add_argument(
        "--config",
        default="config/DINO/DINO_5scale_visdrone_lightweight_100ep.py",
        help="Model config used to build the checkpoint architecture.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth file.")
    parser.add_argument("--image", required=True, help="Input image for a smoke-test inference.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold for printed detections.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    device = torch.device(cli_args.device if cli_args.device == "cpu" or torch.cuda.is_available() else "cpu")

    print(">>> building model from config ...")
    args = SLConfig.fromfile(cli_args.config)
    args.device = str(device)
    args.dataset_file = "coco"  # only affects postprocessor mapping
    model, criterion, postprocessors = build_model_main(args)

    print(">>> loading checkpoint ...")
    ckpt = torch.load(cli_args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"    missing keys   : {len(missing)}")
    print(f"    unexpected keys: {len(unexpected)}")
    if missing:
        print("    first missing  :", missing[:5])
    if unexpected:
        print("    first unexpected:", unexpected[:5])

    model.eval().to(device)

    print(">>> preprocessing image ...")
    image = Image.open(cli_args.image).convert("RGB")
    orig_w, orig_h = image.size
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img, _ = transform(image, None)

    print(">>> running inference ...")
    with torch.no_grad():
        output = model(img[None].to(device))
    target_sizes = torch.tensor([[orig_h, orig_w]], dtype=torch.float32, device=device)
    output = postprocessors["bbox"](output, target_sizes)[0]

    scores = output["scores"]
    labels = output["labels"]
    boxes = output["boxes"]
    keep = scores > cli_args.threshold

    print("\n===== DETECTIONS (score > {:.2f}) =====".format(cli_args.threshold))
    print(f"total above threshold: {keep.sum().item()}")
    for s, l, b in zip(scores[keep], labels[keep], boxes[keep]):
        idx = int(l)
        name = VISDRONE_CLASSES[idx] if idx < len(VISDRONE_CLASSES) else f"cls{idx}"
        coords = [round(float(x), 3) for x in b]
        print(f"  {name:<18} score={float(s):.3f} box(xyxy)={coords}")

    print("\n>>> PyTorch inference OK. Model is ready for ONNX export.")


if __name__ == "__main__":
    main()
