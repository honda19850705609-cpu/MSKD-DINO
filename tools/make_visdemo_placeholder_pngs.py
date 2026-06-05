#!/usr/bin/env python3
"""无 PyTorch：生成若干「示意图」PNG，仅作版式/色标参考；真实预测请用 visdrone_draw_predictions.py。"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def _noise_bg(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(0.55, 0.08, (h, w, 3)).clip(0, 1)
    z[:, :, 1] *= 0.95
    z[:, :, 2] *= 0.9
    return z


# 导出 dpi：过低 + imshow 默认双线性放大小数组会显得「糊」
_EXPORT_DPI = 200


def fig1_legend(out: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 2.2), dpi=_EXPORT_DPI)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.add_patch(
        mpatches.Rectangle((0.2, 0.35), 0.6, 0.35, fill=False, edgecolor="green", linewidth=3)
    )
    ax.text(1.0, 0.52, "GT (green)", fontsize=11, va="center")
    ax.add_patch(
        mpatches.Rectangle((4.5, 0.35), 0.6, 0.35, fill=False, edgecolor="steelblue", linewidth=3)
    )
    ax.text(5.3, 0.52, "Prediction (color + label + score)", fontsize=11, va="center")
    ax.text(
        0.2,
        0.12,
        "Placeholder only. Real images: tools/visdrone_draw_predictions.py",
        fontsize=9,
        color="0.35",
    )
    fig.savefig(out, bbox_inches="tight", dpi=_EXPORT_DPI)
    plt.close(fig)


def fig2_synthetic_scene(out: Path, seed: int) -> None:
    # 与画布像素量级接近，减少 imshow 放大时的插值糊边（仍非真实照片）
    h, w = 720, 1280
    bg = _noise_bg(h, w, seed)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.6), dpi=_EXPORT_DPI)
    # 小数组硬放大时用最近邻，避免双线性把噪声抹成「糊」
    ax.imshow(bg, interpolation="nearest")
    ax.axis("off")
    # 模拟若干 GT（绿）与预测（蓝/橙）（坐标与 h,w=720×1280 一致）
    gt = [(160, 240, 360, 520), (800, 400, 1040, 600), (500, 160, 680, 320)]
    pr = [(170, 250, 350, 510), (790, 390, 1050, 610), (510, 170, 670, 316), (960, 200, 1160, 440)]
    for x1, y1, x2, y2 in gt:
        ax.add_patch(
            mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2.5, alpha=0.9
            )
        )
    colors = ["deepskyblue", "orange", "cyan", "violet"]
    labels = [("car", 0.62), ("car", 0.58), ("pedestrian", 0.41), ("van", 0.33)]
    for k, (x1, y1, x2, y2) in enumerate(pr):
        ax.add_patch(
            mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=colors[k % len(colors)],
                linewidth=2,
                alpha=0.95,
            )
        )
        name, sc = labels[k]
        ax.text(x1, max(4, y1 - 8), f"{name} {sc:.2f}", color=colors[k % len(colors)], fontsize=8)
    ax.set_title("Synthetic demo (NOT model output): preds near GT = good trend", fontsize=10)
    fig.savefig(out, bbox_inches="tight", dpi=_EXPORT_DPI)
    plt.close(fig)


def fig3_chaotic(out: Path) -> None:
    """对比：乱框示意"""
    h, w = 720, 1280
    bg = _noise_bg(h, w, 99)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.6), dpi=_EXPORT_DPI)
    ax.imshow(bg, interpolation="nearest")
    ax.axis("off")
    rng = np.random.default_rng(42)
    for _ in range(35):
        x1, y1 = rng.integers(0, w - 40), rng.integers(0, h - 40)
        x2, y2 = x1 + rng.integers(15, 80), y1 + rng.integers(15, 80)
        ax.add_patch(
            mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=rng.random(3),
                linewidth=1,
                alpha=0.7,
            )
        )
    ax.text(
        10,
        24,
        "Demo: many random boxes (low thresh / early training)",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.45),
    )
    fig.savefig(out, bbox_inches="tight", dpi=_EXPORT_DPI)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate placeholder VisDemo PNGs (no PyTorch).")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/assets/visdemo_placeholders).",
    )
    args = p.parse_args(argv)
    if args.out is not None:
        root = args.out.expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[1] / "assets" / "visdemo_placeholders"
    root.mkdir(parents=True, exist_ok=True)
    fig1_legend(root / "01_legend_gt_vs_pred.png")
    fig2_synthetic_scene(root / "02_synthetic_aligned_example.png", seed=7)
    fig3_chaotic(root / "03_synthetic_chaotic_boxes_example.png")
    readme = root / "README.txt"
    readme.write_text(
        "这些 PNG 为 matplotlib 生成的示意图，不是 DINO 在 VisDrone 上的真实推理结果。\n"
        "要生成真实预测图，请在安装 PyTorch 的环境中运行:\n"
        "  python tools/visdrone_draw_predictions.py --random_sample --draw_gt ...\n",
        encoding="utf-8",
    )
    print(str(root))


if __name__ == "__main__":
    main()
