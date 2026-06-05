#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练 output_dir 的 log.txt 解析 epoch_time、n_parameters 等，绘制「效率」相关图：
每个 epoch 耗时、累计训练时间、AP@0.5 随累计时间变化（无 PyTorch）。

用法:
  python tools/plot_efficiency_from_log.py --result_dir "G:/.../your_exp"
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
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def parse_epoch_time_seconds(s: object) -> float | None:
    """解析 main.py 写入的 epoch_time 字符串，如 '0:01:15' 或 '1:23:45'。"""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
    except ValueError:
        return None
    return None


def plot_efficiency(rows: list[dict], out_dir: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    epochs: list[int] = []
    sec_per_epoch: list[float] = []
    ap50: list[float] = []
    n_params: list[int] = []

    for row in rows:
        ep = row.get("epoch")
        if ep is None:
            continue
        epochs.append(int(ep))
        t = parse_epoch_time_seconds(row.get("epoch_time"))
        sec_per_epoch.append(t if t is not None else float("nan"))
        v = row.get("test_coco_eval_bbox")
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                ap50.append(float(v[1]))
            except (TypeError, ValueError):
                ap50.append(float("nan"))
        else:
            ap50.append(float("nan"))
        np_ = row.get("n_parameters")
        if isinstance(np_, (int, float)):
            n_params.append(int(np_))
        else:
            n_params.append(0)

    if not epochs:
        (out_dir / "efficiency_summary.txt").write_text(
            "log.txt 中无带 epoch 字段的行，无法绘制效率图。\n", encoding="utf-8"
        )
        return

    # --- 每 epoch 耗时（秒）---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(epochs, sec_per_epoch, color="steelblue", alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds / epoch")
    ax.set_title(((title + " — ") if title else "") + "Wall time per epoch (train+val)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "epoch_duration_seconds.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 累计时间（小时）---
    valid_sec = [s for s in sec_per_epoch if s == s and s is not None]
    cum_hours: list[float] = []
    total_s = 0.0
    for s in sec_per_epoch:
        if s == s and s > 0:
            total_s += s
        cum_hours.append(total_s / 3600.0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, cum_hours, "b-o", ms=4, lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cumulative hours")
    ax.set_title(((title + " — ") if title else "") + "Cumulative training wall time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "cumulative_hours_vs_epoch.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- AP@0.5 vs 累计时间（小时）---
    if any(x == x for x in ap50):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(cum_hours, ap50, "g-s", ms=4, lw=1.5)
        ax.set_xlabel("cumulative wall time (hours)")
        ax.set_ylabel("validation AP@0.5")
        ax.set_title(((title + " — ") if title else "") + "AP@0.5 vs cumulative time (efficiency)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_dir / "ap50_vs_cumulative_hours.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Summary ---
    mean_s = sum(valid_sec) / len(valid_sec) if valid_sec else float("nan")
    total_h = total_s / 3600.0
    lines = [
        "Training efficiency summary (from log.txt)",
        f"epochs logged: {len(epochs)} (epoch index {min(epochs)} … {max(epochs)})",
        f"mean wall time per epoch: {mean_s:.1f} s" if mean_s == mean_s else "mean wall time: n/a",
        f"total wall time (sum of epoch_time): {total_h:.3f} h ({total_s / 60:.1f} min)",
        "",
    ]
    if n_params and any(n > 0 for n in n_params):
        lines.append(f"n_parameters (last row): {n_params[-1]:,}")
    if mean_s == mean_s and len(epochs) > 0:
        lines.append(
            f"rough extrapolation to 100 epochs: {mean_s * 100 / 3600:.2f} h wall time (linear, same setup)"
        )
    lines.append("")
    lines.append("Figures: epoch_duration_seconds.png, cumulative_hours_vs_epoch.png, ap50_vs_cumulative_hours.png")
    (out_dir / "efficiency_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    readme = out_dir / "README.txt"
    readme.write_text(
        "本目录由 tools/plot_efficiency_from_log.py 生成。\n"
        "数据来自训练 log.txt 中的 epoch_time、test_coco_eval_bbox。\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot training time / efficiency from log.txt")
    ap.add_argument("--result_dir", type=str, required=True, help="训练 output_dir（含 log.txt）")
    ap.add_argument(
        "--out_subdir",
        type=str,
        default="viz_efficiency",
        help="在 result_dir 下创建的子文件夹名",
    )
    ap.add_argument("--title", type=str, default="", help="图表标题前缀")
    args = ap.parse_args()

    result_dir = Path(args.result_dir).expanduser().resolve()
    out_dir = result_dir / args.out_subdir
    rows = _read_jsonl_log(result_dir / "log.txt")
    plot_efficiency(rows, out_dir, title=args.title or result_dir.name)
    print(f"Efficiency plots saved under:\n  {out_dir}")


if __name__ == "__main__":
    main()
