#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 DINO 训练输出目录的 log.txt（JSON 行）与 info.txt 解析指标并绘图。
仅依赖 matplotlib + 标准库，不需要 PyTorch。

用法:
  python tools/plot_metrics_from_logs.py --result_dir "G:/.../your_exp"
  python tools/plot_metrics_from_logs.py --result_dir . --out_subdir metrics_plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl_log(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not log_path.is_file():
        return rows
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _coco_stats(row: dict, prefix: str) -> tuple[float | None, float | None]:
    """prefix: 'test_' or 'ema_test_'"""
    key = f"{prefix}coco_eval_bbox"
    v = row.get(key)
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None, None
    try:
        return float(v[0]), float(v[1])
    except (TypeError, ValueError):
        return None, None


def _coco_stats_list(row: dict, prefix: str) -> list[float] | None:
    key = f"{prefix}coco_eval_bbox"
    v = row.get(key)
    if not isinstance(v, (list, tuple)) or len(v) < 6:
        return None
    try:
        return [float(x) for x in v[:12]]
    except (TypeError, ValueError):
        return None


def _extract_train_loss(row: dict) -> float | None:
    for k in ("train_loss", "train_total_loss"):
        if k in row and isinstance(row[k], (int, float)):
            return float(row[k])
    return None


def _extract_lr(row: dict) -> float | None:
    v = row.get("train_lr")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def plot_from_rows(rows: list[dict], out_dir: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    epochs: list[int] = []
    ap: list[float] = []
    ap50: list[float] = []
    ap_ema: list[float | None] = []
    ap50_ema: list[float | None] = []
    loss: list[float | None] = []
    lr: list[float | None] = []

    for row in rows:
        ep = row.get("epoch")
        if ep is None:
            continue
        epochs.append(int(ep))
        m, m50 = _coco_stats(row, "test_")
        ap.append(m if m is not None else float("nan"))
        ap50.append(m50 if m50 is not None else float("nan"))
        em, em50 = _coco_stats(row, "ema_test_")
        ap_ema.append(em)
        ap50_ema.append(em50)
        loss.append(_extract_train_loss(row))
        lr.append(_extract_lr(row))

    if not epochs:
        (out_dir / "metrics_summary.txt").write_text(
            "未从 log.txt 解析到任何带 epoch 字段的 JSON 行。\n"
            "请确认 output_dir 下有 log.txt，且为训练时追加写入的 JSONL。\n",
            encoding="utf-8",
        )
        return

    # --- Figure: mAP ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, ap, "b-o", ms=3, lw=1.5, label="bbox mAP (0.5:0.95)")
    ax.plot(epochs, ap50, "g-s", ms=3, lw=1.5, label="AP@0.5")
    if any(x is not None for x in ap_ema):
        em = [x if x is not None else float("nan") for x in ap_ema]
        ax.plot(epochs, em, "c--^", ms=3, lw=1.2, label="EMA mAP (if any)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("AP")
    ax.set_title(title or "Validation COCO bbox AP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, out_dir / "map_ap50_vs_epoch.png")
    plt.close(fig)

    # --- APs / APm / APl (COCO bbox stats index 3,4,5) ---
    aps_list: list[float | None] = []
    apm_list: list[float | None] = []
    apl_list: list[float | None] = []
    for row in rows:
        ep = row.get("epoch")
        if ep is None:
            continue
        st = _coco_stats_list(row, "test_")
        if st is None or len(st) < 6:
            aps_list.append(None)
            apm_list.append(None)
            apl_list.append(None)
        else:
            aps_list.append(st[3])
            apm_list.append(st[4])
            apl_list.append(st[5])
    if epochs and any(x is not None for x in aps_list + apm_list + apl_list):
        fig, ax = plt.subplots(figsize=(9, 5))
        def _line(vals: list[float | None], label: str, color: str, marker: str):
            y = [v if v is not None else float("nan") for v in vals]
            ax.plot(epochs, y, color=color, marker=marker, ms=3, lw=1.5, label=label)
        _line(aps_list, "AP small", "#1f77b4", "o")
        _line(apm_list, "AP medium", "#d62728", "s")
        _line(apl_list, "AP large", "#9467bd", "^")
        ax.set_xlabel("epoch")
        ax.set_ylabel("AP")
        ax.set_title(((title + " — ") if title else "") + "COCO AP by area (bbox)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _save_fig(fig, out_dir / "aps_apm_apl_vs_epoch.png")
        plt.close(fig)

    # --- Loss ---
    if any(x is not None for x in loss):
        fig, ax = plt.subplots(figsize=(9, 5))
        le = [x if x is not None else float("nan") for x in loss]
        ax.plot(epochs, le, "r-", lw=1.5, label="train loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(title or "Training loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _save_fig(fig, out_dir / "train_loss_vs_epoch.png")
        plt.close(fig)

    # --- LR ---
    if any(x is not None for x in lr):
        fig, ax = plt.subplots(figsize=(9, 4))
        lv = [x if x is not None else float("nan") for x in lr]
        ax.plot(epochs, lv, color="purple", lw=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("lr")
        ax.set_title(title or "Learning rate")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save_fig(fig, out_dir / "lr_vs_epoch.png")
        plt.close(fig)

    # --- Summary text ---
    valid_ap = [(i, ap[i]) for i in range(len(ap)) if ap[i] == ap[i]]
    best_i = max(valid_ap, key=lambda t: t[1])[0] if valid_ap else 0
    lines = [
        "Metrics summary (from log.txt JSON lines)",
        f"epochs parsed: {len(epochs)}",
        f"last epoch: {epochs[-1]}",
        "",
    ]
    if valid_ap:
        lines.extend(
            [
                f"best test mAP (0.5:0.95): {ap[best_i]:.4f} @ epoch {epochs[best_i]}",
                f"AP@0.5 at same row: {ap50[best_i]:.4f}",
                "",
            ]
        )
    else:
        lines.extend(["(no finite test_coco_eval_bbox in log)", ""])
    if any(x is not None for x in ap_ema):
        lines.append("(EMA curves plotted if present in log)")
    (out_dir / "metrics_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _save_fig(fig, path: Path) -> None:
    path = Path(path)
    try:
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
    except OSError as e:
        raise SystemExit(f"无法写入图表文件 {path}: {e}") from e


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mAP/loss from DINO log.txt (no PyTorch)")
    ap.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="训练 output_dir（内含 log.txt，可选 info.txt）",
    )
    ap.add_argument(
        "--out_subdir",
        type=str,
        default="metrics_plots_from_log",
        help="在 result_dir 下创建的子文件夹名",
    )
    ap.add_argument("--title", type=str, default="", help="图表标题前缀")
    args = ap.parse_args()

    result_dir = Path(args.result_dir).expanduser().resolve()
    out_dir = result_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = result_dir / "log.txt"
    rows = _read_jsonl_log(log_path)
    plot_from_rows(rows, out_dir, title=args.title or result_dir.name)

    readme = out_dir / "README.txt"
    readme.write_text(
        "本目录由 tools/plot_metrics_from_logs.py 生成。\n"
        f"数据源: {log_path}\n"
        "图表: map_ap50_vs_epoch.png, aps_apm_apl_vs_epoch.png, train_loss_vs_epoch.png（若有）, lr_vs_epoch.png（若有）\n",
        encoding="utf-8",
    )
    print(f"Done. Plots saved under:\n  {out_dir}")


if __name__ == "__main__":
    main()
