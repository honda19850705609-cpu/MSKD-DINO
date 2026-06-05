"""DINO-DETR: load checkpoint and run one PyTorch inference (Day 3).

Verifies the model can be built with the training config, load the VisDrone
checkpoint, and run a forward pass — before attempting ONNX export.

Run from the project root:
    cd /root/autodl-tmp/Day3/DINO-DETR_improve_two_for_all_train_April_thirteenth_lighting
    python verify_inference.py
"""

import os
import torch
from PIL import Image

from main import build_model_main
from util.slconfig import SLConfig
import datasets.transforms as T

# -------- paths (edit if needed) --------
CONFIG_PATH = "config/DINO/DINO_5scale_visdrone_lightweight_100ep.py"
CKPT_PATH = "/root/autodl-tmp/Day3/checkpoint_best_ema.pth"
IMAGE_PATH = "/root/autodl-tmp/Day3/test.jpg"
THRESHOLD = 0.3

# VisDrone 10 classes (index 0..9)
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def main():
    print(">>> building model from config ...")
    args = SLConfig.fromfile(CONFIG_PATH)
    args.device = "cuda"
    args.dataset_file = "coco"  # only affects postprocessor mapping
    model, criterion, postprocessors = build_model_main(args)

    print(">>> loading checkpoint ...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"    missing keys   : {len(missing)}")
    print(f"    unexpected keys: {len(unexpected)}")
    if missing:
        print("    first missing  :", missing[:5])
    if unexpected:
        print("    first unexpected:", unexpected[:5])

    model.eval().cuda()

    print(">>> preprocessing image ...")
    image = Image.open(IMAGE_PATH).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img, _ = transform(image, None)

    print(">>> running inference ...")
    with torch.no_grad():
        output = model(img[None].cuda())
    # postprocess: needs original target sizes (h, w); use normalized [1,1]
    output = postprocessors["bbox"](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    scores = output["scores"]
    labels = output["labels"]
    boxes = output["boxes"]
    keep = scores > THRESHOLD

    print("\n===== DETECTIONS (score > {:.1f}) =====".format(THRESHOLD))
    print(f"total above threshold: {keep.sum().item()}")
    for s, l, b in zip(scores[keep], labels[keep], boxes[keep]):
        idx = int(l)
        name = VISDRONE_CLASSES[idx] if idx < len(VISDRONE_CLASSES) else f"cls{idx}"
        coords = [round(float(x), 3) for x in b]
        print(f"  {name:<18} score={float(s):.3f} box(cxcywh-norm)={coords}")

    print("\n>>> PyTorch inference OK. Model is ready for ONNX export.")


if __name__ == "__main__":
    main()
