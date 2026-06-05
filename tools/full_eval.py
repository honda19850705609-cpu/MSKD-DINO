"""
Full evaluation helper.

This script is a thin wrapper around main.py --eval, so you can evaluate a checkpoint
without retyping config/dataset arguments.

Example:
  python tools/full_eval.py ^
    --resume outputs/final_stretch/checkpoint_best_regular.pth ^
    --config_file config/DINO/DINO_5scale_visdrone_final.py ^
    --val_ann_file path/to/instances_val2017.json ^
    --val_img_folder path/to/images ^
    --amp
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("full_eval")
    p.add_argument("--resume", required=True, help="Checkpoint .pth to evaluate.")
    p.add_argument(
        "--config_file",
        "-c",
        default=str(_repo_root() / "config" / "DINO" / "DINO_5scale_visdrone_final.py"),
        help="Config file used for evaluation (must match training).",
    )
    p.add_argument("--val_img_folder", default=None, help="Val images folder.")
    p.add_argument("--val_ann_file", default=None, help="Val COCO JSON file.")
    p.add_argument("--dataset_json_root", default="", help="Alternative dataset root (contains annotations/...).")
    p.add_argument("--dataset_images", default="", help="Alternative images folder (pair with dataset_json_root).")
    p.add_argument("--coco_path", default="", help="Standard COCO root (train2017/val2017/annotations).")
    p.add_argument("--output_dir", default="", help="Where to write eval logs (optional).")
    p.add_argument("--amp", action="store_true", help="Enable AMP during eval.")
    p.add_argument(
        "--extra_options",
        nargs="*",
        default=[],
        help="Extra --options overrides (key=value). Example: num_select=500",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = _repo_root()

    # Ensure imports resolve like running from repo root.
    os.chdir(str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Delegate to main.py CLI to avoid duplicating evaluation logic.
    cmd = [
        sys.executable,
        "main.py",
        "--eval",
        "--resume",
        str(args.resume),
        "--config_file",
        str(args.config_file),
    ]
    if args.output_dir:
        cmd += ["--output_dir", str(args.output_dir)]
    if args.amp:
        cmd += ["--amp"]

    # Dataset args (pass-through; user may provide any one of these styles).
    if args.val_img_folder:
        cmd += ["--val_img_folder", str(args.val_img_folder)]
    if args.val_ann_file:
        cmd += ["--val_ann_file", str(args.val_ann_file)]
    if args.dataset_json_root:
        cmd += ["--dataset_json_root", str(args.dataset_json_root)]
    if args.dataset_images:
        cmd += ["--dataset_images", str(args.dataset_images)]
    if args.coco_path:
        cmd += ["--coco_path", str(args.coco_path)]

    if args.extra_options:
        cmd += ["--options"] + list(args.extra_options)

    # Run.
    import subprocess

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键评测 + 可视化：四个路径 → 指标 JSON、COCO 结果、终端报告、9 张图。

本文件位于训练仓库 tools/ 下，与根目录 main.py（训练）分离，请勿覆盖训练入口。

在仓库根目录执行（须能 import util、datasets、根目录 main.build_model_main）:
  python tools/visdrone_full_eval.py -i
  python tools/visdrone_full_eval.py --resume ... --val_ann_file ... --val_img_folder ...

若脚本被单独拷走、旁无 util/，请设 DINO_ROOT 或 --dino_root 指向本训练仓库根目录。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
# VisDrone 常量
# -----------------------------------------------------------------------------
VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
CORE_INDICES = [0, 1, 2, 3, 4, 5, 8]
NUM_CLASSES = 10

# 用户无需在界面填写的固定默认
DEFAULT_MAX_DETS = 500
DEFAULT_MODEL_NAME = "DINO-VisDrone"


def _repo_root() -> Path:
    """本脚本所在目录及上一级：用于推断是否「脚本与 DINO 同仓库」。"""
    p = Path(__file__).resolve().parent
    if (p / "util").is_dir():
        return p
    if (p.parent / "util").is_dir():
        return p.parent
    return p


def resolve_dino_root(dino_root_arg: str) -> Path:
    """含 util/、datasets/ 的 DINO 源码根。优先 --dino_root / 环境变量，再尝试脚本旁目录。"""
    tried: List[str] = []
    for raw in (
        (dino_root_arg or "").strip(),
        (os.environ.get("DINO_ROOT") or "").strip(),
        (os.environ.get("VISDRONE_DINO_ROOT") or "").strip(),
    ):
        if not raw:
            continue
        tried.append(raw)
        p = Path(raw).expanduser().resolve()
        if (p / "util").is_dir():
            return p

    inferred = _repo_root().resolve()
    if (inferred / "util").is_dir():
        return inferred

    hint = (
        "未找到含 util/ 的 DINO 源码目录（本脚本旁也没有）。\n\n"
        "本文件夹若只有评测脚本、没有训练时的完整工程，请任选其一：\n"
        "  1) 把训练仓库里整个 DINO 目录（含 util、datasets、models 等）拷到本机；\n"
        "  2) Colab: os.environ['DINO_ROOT']='/content/drive/.../你的DINO仓库根'\n"
        "  3) 命令行: python tools/visdrone_full_eval.py --dino_root /path/to/DINO --resume ...\n\n"
        f"已检查的路径: {tried or ['(未设置 DINO_ROOT / --dino_root)']}\n"
        f"脚本旁推断目录: {inferred}"
    )
    raise FileNotFoundError(hint)


def ensure_repo_on_path(dino_root_arg: str = "") -> None:
    """把 DINO 仓库根放在 sys.path 最前，保证 import util / datasets / main 来自训练代码。"""
    root = resolve_dino_root(dino_root_arg)
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)


# 任务书阈值（图 9 与终端对比用）
THESIS_BASE = {
    "core_p": 0.70,
    "core_r": 0.75,
    "core_f1": 0.72,
    "all_p": 0.65,
    "all_r": 0.70,
    "all_f1": 0.67,
    "ap50": 0.35,
    "aps": 0.20,
    "fnr": 0.25,
    "fpr": 0.20,
}
THESIS_STRETCH = {
    "core_p": 0.80,
    "core_r": 0.85,
    "core_f1": 0.82,
    "all_p": 0.75,
    "all_r": 0.80,
    "all_f1": 0.77,
    "ap50": 0.55,
    "aps": 0.30,
    "fnr": 0.15,
    "fpr": 0.10,
}


def _apply_interactive_dict(args: argparse.Namespace, data: Dict[str, Any]) -> None:
    """交互收集到的路径写入 args（仅非空项）。"""
    for k, v in data.items():
        if not hasattr(args, k):
            continue
        if v is None or v == "":
            continue
        setattr(args, k, v)


def _attach_eval_defaults(args: argparse.Namespace) -> None:
    """评测内部固定项：config / 设备 / max_dets / 报告名等由程序自动处理，不提供命令行开关。"""
    args.config_file = ""
    args.device = ""
    args.max_dets = DEFAULT_MAX_DETS
    args.model_name = DEFAULT_MODEL_NAME
    args.det_json = ""


def run_interactive_gui() -> Optional[Dict[str, Any]]:
    """图形界面：仅四项路径。取消返回 None；无显示器时返回 None。"""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        return None

    out: Dict[str, Any] = {}
    cancelled = {"v": True}

    try:
        root = tk.Tk()
    except tk.TclError:
        # 常见：Colab / 无 $DISPLAY 的 Linux / 纯 SSH
        return None
    root.title("一键评测 — 路径")
    root.minsize(680, 320)
    root.resizable(True, True)

    main = ttk.Frame(root, padding=10)
    main.pack(fill=tk.BOTH, expand=True)

    rows: List[Tuple[str, str, str]] = [
        ("resume", "① 权重 checkpoint（.pth）", "file"),
        ("val_ann_file", "② 验证集标注（COCO JSON）", "file_json"),
        ("val_img_folder", "③ 验证集图片文件夹", "dir"),
        ("output_dir", "④ 输出目录（可空=自动）", "dir_opt"),
        ("dino_root", "⑤ DINO 源码根（须含 util/，可空=本目录或环境变量 DINO_ROOT）", "dir_opt"),
    ]

    vars_map: Dict[str, tk.StringVar] = {}
    r = 0
    for key, label, mode in rows:
        ttk.Label(main, text=label).grid(row=r, column=0, sticky=tk.W, pady=3)
        v = tk.StringVar()
        vars_map[key] = v
        ent = ttk.Entry(main, textvariable=v, width=64)
        ent.grid(row=r, column=1, sticky=tk.EW, padx=6, pady=3)

        def make_browse(m=mode, var=v):
            def browse():
                p = ""
                if m == "file":
                    p = filedialog.askopenfilename(
                        title="选择 checkpoint",
                        filetypes=[("PyTorch", "*.pth *.pt"), ("All", "*.*")],
                    )
                elif m == "file_json":
                    p = filedialog.askopenfilename(
                        title="选择 JSON",
                        filetypes=[("JSON", "*.json"), ("All", "*.*")],
                    )
                elif m in ("dir", "dir_opt"):
                    p = filedialog.askdirectory(title="选择文件夹")
                else:
                    return
                if p:
                    var.set(p)

            return browse

        ttk.Button(main, text="浏览…", width=8, command=make_browse()).grid(row=r, column=2, pady=3)
        r += 1

    main.columnconfigure(1, weight=1)

    hint = ttk.Label(
        main,
        text="①②③ 必填；④⑤ 可空。若本目录无 util/，⑤ 请选训练仓库根目录或设 DINO_ROOT。",
        foreground="#444",
    )
    hint.grid(row=r, column=0, columnspan=3, sticky=tk.W, pady=(8, 4))
    r += 1

    btn_row = ttk.Frame(main)
    btn_row.grid(row=r, column=0, columnspan=3, pady=12)

    def on_ok():
        resume = vars_map["resume"].get().strip()
        ann = vars_map["val_ann_file"].get().strip()
        imgd = vars_map["val_img_folder"].get().strip()
        if not resume or not ann or not imgd:
            messagebox.showwarning("缺少必填项", "请填写 ① 权重、② 标注 JSON、③ 图片文件夹。")
            return
        out["resume"] = resume
        out["val_ann_file"] = ann
        out["val_img_folder"] = imgd
        out["output_dir"] = vars_map["output_dir"].get().strip()
        out["dino_root"] = vars_map["dino_root"].get().strip()
        cancelled["v"] = False
        root.destroy()

    def on_cancel():
        root.destroy()

    ttk.Button(btn_row, text="开始评测", command=on_ok).pack(side=tk.LEFT, padx=6)
    ttk.Button(btn_row, text="取消", command=on_cancel).pack(side=tk.LEFT, padx=6)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()

    if cancelled["v"]:
        return None
    return out


def run_interactive_cli() -> Optional[Dict[str, Any]]:
    """无图形界面时，终端只问四个路径。"""
    print("=== 一键评测 — 终端输入（④ 可直接回车留空）===\n")

    def ask_line(prompt: str, default: str = "") -> str:
        tip = f" [{default}]" if default else ""
        s = input(f"{prompt}{tip}: ").strip().strip('"')
        return s if s else default

    resume = ask_line("① checkpoint .pth")
    ann = ask_line("② 验证集 COCO JSON")
    imgd = ask_line("③ 验证集图片文件夹")
    if not resume or not ann or not imgd:
        print("错误：①②③ 为必填。")
        return None

    out_dir = ask_line("④ 输出目录（可空=自动）")
    droot = ask_line("⑤ DINO 源码根含 util/（可空，或用环境变量 DINO_ROOT）")

    return {
        "resume": resume,
        "val_ann_file": ann,
        "val_img_folder": imgd,
        "output_dir": out_dir,
        "dino_root": droot,
    }


def run_interactive_mode() -> Optional[Dict[str, Any]]:
    data = run_interactive_gui()
    if data is not None:
        return data
    print(
        "无法使用图形窗口（无显示器 / Colab 等），改用终端逐项输入路径。\n",
        file=sys.stderr,
    )
    return run_interactive_cli()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DINO VisDrone 一键评测与可视化（仅需四个路径；config/GPU 等自动处理）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="交互输入四个路径（优先图形窗口，否则终端）",
    )
    p.add_argument("--resume", default="", help="① checkpoint .pth")
    p.add_argument("--val_ann_file", default="", help="② 验证集 COCO JSON")
    p.add_argument("--val_img_folder", default="", help="③ 验证集图片目录")
    p.add_argument("--output_dir", default="", help="④ 输出目录；空=权重同目录下自动 eval_*")
    p.add_argument(
        "--dino_root",
        default="",
        help="训练时的 DINO 仓库根目录（须含 util/）。默认同级查找；也可设环境变量 DINO_ROOT",
    )
    args = p.parse_args()

    _attach_eval_defaults(args)

    if args.interactive:
        data = run_interactive_mode()
        if data is None:
            sys.exit(0)
        _apply_interactive_dict(args, data)

    # 未带齐前三项时自动进入交互，便于直接 python tools/visdrone_full_eval.py
    if not args.resume or not args.val_ann_file or not args.val_img_folder:
        if not args.interactive:
            print(
                "未提供完整的 --resume / --val_ann_file / --val_img_folder，进入交互输入。\n",
                file=sys.stderr,
            )
        data = run_interactive_mode()
        if data is None:
            sys.exit(0)
        _apply_interactive_dict(args, data)

    if not args.resume or not args.val_ann_file or not args.val_img_folder:
        p.error("仍缺少 --resume、--val_ann_file 或 --val_img_folder。")
    return args


def _args_get(saved: Union[object, Dict], key: str, default: str = "") -> str:
    if isinstance(saved, dict):
        return str(saved.get(key, default) or default)
    return str(getattr(saved, key, default) or default)


def load_checkpoint_paths(resume: str) -> Tuple[dict, Any, str]:
    """返回 (ckpt dict, model_state_dict, resolved_config_file)."""
    import torch

    resume_path = Path(resume)
    # PyTorch 2.6+ 默认 weights_only=True，含 argparse.Namespace 的旧 checkpoint 会失败
    try:
        ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(resume_path), map_location="cpu")
    config_file = ""

    if "args" in ckpt:
        saved = ckpt["args"]
        config_file = _args_get(saved, "config_file")

    stem_lower = resume_path.stem.lower()
    if "ema" in stem_lower:
        model_state = ckpt.get("ema_model", ckpt.get("model"))
    else:
        model_state = ckpt.get("model", ckpt)

    if not config_file:
        cand = resume_path.parent / "config_cfg.py"
        if cand.is_file():
            config_file = str(cand)
    return ckpt, model_state, config_file


def ensure_config_file(args: argparse.Namespace, config_from_ckpt: str) -> str:
    if args.config_file:
        return args.config_file
    if config_from_ckpt:
        return config_from_ckpt
    raise FileNotFoundError(
        "无法确定 config_file。请用 --config_file 指定，或在 checkpoint 同目录放置 config_cfg.py。"
    )


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        stem = Path(args.resume).stem
        out = Path(args.resume).parent / f"eval_{stem}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    return out


def resolve_device(device_str: str):
    """未指定时自动选择：CUDA GPU > Apple MPS > CPU，并打印到 stderr。"""
    import torch

    if device_str:
        dev = torch.device(device_str)
    elif torch.cuda.is_available():
        dev = torch.device("cuda:0")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else 0
        try:
            name = torch.cuda.get_device_name(idx)
            cnt = torch.cuda.device_count()
            print(f"[设备] CUDA：{dev} | {name}（共 {cnt} 张 GPU）", file=sys.stderr)
        except Exception:
            print(f"[设备] CUDA：{dev}", file=sys.stderr)
    elif dev.type == "mps":
        print("[设备] Apple Silicon MPS", file=sys.stderr)
    else:
        print("[设备] 未检测到 GPU，使用 CPU", file=sys.stderr)
    return dev


# --- IoU & 贪心匹配 ---
def box_xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match_one_image_cat(
    dets_xywh: np.ndarray,
    det_scores: np.ndarray,
    gts_xywh: np.ndarray,
    iou_thr: float = 0.5,
) -> Tuple[int, int, int]:
    """返回 tp, fp, fn。det 按 score 降序。"""
    if dets_xywh.size == 0:
        return 0, 0, len(gts_xywh)
    if gts_xywh.size == 0:
        return 0, len(dets_xywh), 0
    gts_xyxy = np.stack([box_xywh_to_xyxy(g) for g in gts_xywh], axis=0)
    gt_matched = np.zeros(len(gts_xyxy), dtype=bool)
    tp = fp = 0
    for i in range(len(dets_xywh)):
        d_xyxy = box_xywh_to_xyxy(dets_xywh[i])
        best_j = -1
        best_iou = 0.0
        for j in range(len(gts_xyxy)):
            if gt_matched[j]:
                continue
            iou = iou_xyxy(d_xyxy, gts_xyxy[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            gt_matched[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = int((~gt_matched).sum())
    return tp, fp, fn


def load_coco_gt(ann_path: str):
    from pycocotools.coco import COCO

    return COCO(ann_path)


def build_gt_index(coco_gt) -> Dict[Tuple[int, int], List[Dict]]:
    """(image_id, cat_id) -> list of ann dict (bbox xywh)."""
    idx: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for aid, ann in coco_gt.anns.items():
        k = (ann["image_id"], ann["category_id"])
        idx[k].append(ann)
    return idx


def compute_prf1_at_thresholds(
    coco_results: List[dict],
    coco_gt,
    thresholds: Sequence[float],
) -> Dict[str, Any]:
    """贪心 IoU>=0.5，返回每个阈值下全类/核心类/每类 P/R/F1/FNR/FPR。"""
    gt_index = build_gt_index(coco_gt)
    # 按 (image_id, cat_id) 分组检测
    dets_by_key: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for r in coco_results:
        k = (r["image_id"], r["category_id"])
        dets_by_key[k].append(r)
    # 含 GT 但无预测的 (image, cat) 也要参与（否则漏检未计入）
    all_keys = set(gt_index.keys()) | set(dets_by_key.keys())

    sweep: Dict[str, Any] = {}
    best_all = {"f1": -1.0, "thr": None, "p": 0.0, "r": 0.0}
    best_core = {"f1": -1.0, "thr": None, "p": 0.0, "r": 0.0}

    for thr in thresholds:
        # 全类 / 核心 / 每类 累计
        tp_a = fp_a = fn_a = 0
        tp_c = fp_c = fn_c = 0
        per_cat = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in range(NUM_CLASSES)}

        for key in all_keys:
            dets = dets_by_key.get(key, [])
            gts = gt_index.get(key, [])
            cat_id = key[1]
            gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float64) if gts else np.zeros((0, 4))
            filt = [d for d in dets if d["score"] >= thr]
            filt.sort(key=lambda x: -x["score"])
            det_boxes = np.array([d["bbox"] for d in filt], dtype=np.float64)
            det_scores = np.array([d["score"] for d in filt], dtype=np.float64)

            tp, fp, fn = greedy_match_one_image_cat(det_boxes, det_scores, gt_boxes)
            tp_a += tp
            fp_a += fp
            fn_a += fn
            per_cat[cat_id]["tp"] += tp
            per_cat[cat_id]["fp"] += fp
            per_cat[cat_id]["fn"] += fn
            if cat_id in CORE_INDICES:
                tp_c += tp
                fp_c += fp
                fn_c += fn

        def prf(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0  # 与常见定义：FP/(FP+TP) 在检测里可作 proxy
            return p, r, f1, fnr, fpr

        p_a, r_a, f1_a, fnr_a, fpr_a = prf(tp_a, fp_a, fn_a)
        p_c, r_c, f1_c, fnr_c, fpr_c = prf(tp_c, fp_c, fn_c)
        per_class = {}
        for cid in range(NUM_CLASSES):
            t, f, n = per_cat[cid]["tp"], per_cat[cid]["fp"], per_cat[cid]["fn"]
            pc, rc, f1c, _, _ = prf(t, f, n)
            per_class[str(cid)] = {
                "name": VISDRONE_NAMES[cid],
                "tp": int(t),
                "fp": int(f),
                "fn": int(n),
                "precision": pc,
                "recall": rc,
                "f1": f1c,
            }

        sweep[str(thr)] = {
            "all": {
                "precision": p_a,
                "recall": r_a,
                "f1": f1_a,
                "fnr": fnr_a,
                "fpr": fpr_a,
                "tp": int(tp_a),
                "fp": int(fp_a),
                "fn": int(fn_a),
            },
            "core": {
                "precision": p_c,
                "recall": r_c,
                "f1": f1_c,
                "fnr": fnr_c,
                "fpr": fpr_c,
                "tp": int(tp_c),
                "fp": int(fp_c),
                "fn": int(fn_c),
            },
            "per_class": per_class,
        }
        if f1_a > best_all["f1"]:
            best_all = {"f1": f1_a, "thr": thr, "p": p_a, "r": r_a}
        if f1_c > best_core["f1"]:
            best_core = {"f1": f1_c, "thr": thr, "p": p_c, "r": r_c}

    return {
        "threshold_sweep": sweep,
        "best_all": best_all,
        "best_core": best_core,
    }


def run_coco_eval(ann_path: str, res_path: str, max_dets: int) -> Tuple[List[float], np.ndarray, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_path)
    coco_dt = coco_gt.loadRes(res_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = [1, 10, max_dets]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = list(coco_eval.stats)
    precision = coco_eval.eval["precision"]
    return stats, precision, coco_eval


def _nanmean_valid(x: np.ndarray) -> float:
    m = x[x > -1]
    return float(m.mean()) if m.size else 0.0


def per_class_ap_from_precision(precision: np.ndarray) -> Dict[str, Dict[str, float]]:
    """precision [T,R,K,A,M] — 与文档一致取 AP@0.5、APs、全 AP。"""
    out: Dict[str, Dict[str, float]] = {}
    for k in range(min(NUM_CLASSES, precision.shape[2])):
        ap50 = _nanmean_valid(precision[0, :, k, 0, -1]) if precision.size else 0.0
        aps = _nanmean_valid(precision[:, :, k, 1, -1]) if precision.size else 0.0
        ap = _nanmean_valid(precision[:, :, k, 0, -1]) if precision.size else 0.0
        out[str(k)] = {"name": VISDRONE_NAMES[k], "AP50": ap50, "APs": aps, "AP": ap}
    return out


def build_dataset_val(args):
    from datasets import build_dataset as _bd

    return _bd(image_set="val", args=args)


def run_inference_and_save(
    args: argparse.Namespace,
    model_state: Any,
    config_file: str,
    out_dir: Path,
    device,
) -> str:
    import torch
    from torch.utils.data import DataLoader

    ensure_repo_on_path(getattr(args, "dino_root", "") or "")
    from util.slconfig import SLConfig
    from main import build_model_main
    import util.misc as utils

    cfg = SLConfig.fromfile(config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    for k, v in cfg_dict.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    args.dataset_file = "coco"
    args.train_img_folder = args.val_img_folder
    args.train_ann_file = args.val_ann_file
    args.fix_size = False
    args.masks = False
    args.frozen_weights = None

    model, criterion, postprocessors = build_model_main(args)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()

    dataset_val = build_dataset_val(args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader = DataLoader(
        dataset_val,
        batch_size=1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=4,
    )

    results: List[dict] = []
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            res = postprocessors["bbox"](outputs, orig_target_sizes)[0]

            scores = res["scores"].cpu().numpy()
            labels = res["labels"].cpu().numpy()
            boxes = res["boxes"].cpu().numpy()  # cxcywh normalized? Usually xyxy abs — project-dependent

            image_id = int(targets[0]["image_id"].item())
            # 若 boxes 为 xyxy 像素：
            h, w = targets[0]["orig_size"].tolist()
            for sc, la, box in zip(scores, labels, boxes):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(la),
                        "bbox": [float(x1), float(y1), float(bw), float(bh)],
                        "score": float(sc),
                    }
                )

    det_path = out_dir / "coco_instances_results.json"
    with open(det_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    return str(det_path)


def plot_all(
    out_dir: Path,
    stats: List[float],
    per_class_ap: Dict[str, Dict[str, float]],
    thesis_data: Dict[str, Any],
    model_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out_dir / "plots"
    # 01
    names12 = [
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#1f4e79" if i < 6 else "#c55a11" for i in range(12)]
    x = np.arange(12)
    ax.bar(x, stats[:12], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names12, rotation=45, ha="right")
    ax.set_ylabel("value")
    ax.set_title("COCO 12 metrics")
    for i, v in enumerate(stats[:12]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "01_coco_12_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 02
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].barh(names12[:6], stats[:6], color="#1f4e79")
    axes[0].set_title("AP group")
    axes[1].barh(names12[6:], stats[6:12], color="#c55a11")
    axes[1].set_title("AR group")
    fig.tight_layout()
    fig.savefig(plots / "02_ap_vs_ar_groups.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 03
    fig, ax = plt.subplots(figsize=(6, 4))
    labs = ["Small", "Medium", "Large"]
    ap_sz = [stats[3], stats[4], stats[5]]
    ar_sz = [stats[9], stats[10], stats[11]]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, ap_sz, w, label="AP", color="#1f4e79")
    ax.bar(x + w / 2, ar_sz, w, label="AR", color="#c55a11")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.legend()
    ax.set_title("AP/AR by size")
    fig.tight_layout()
    fig.savefig(plots / "03_ap_ar_by_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 04
    fig, ax = plt.subplots(figsize=(10, 4))
    cats = [per_class_ap[str(i)]["name"] for i in range(NUM_CLASSES)]
    ap50s = [per_class_ap[str(i)]["AP50"] for i in range(NUM_CLASSES)]
    col = ["#1f4e79" if i in CORE_INDICES else "#a6a6a6" for i in range(NUM_CLASSES)]
    ax.bar(cats, ap50s, color=col)
    ax.set_ylabel("AP@0.5")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(plots / "04_per_class_ap50.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 05 radar (6 axes)
    labels_r = ["AP", "AP50", "AP75", "APs", "AR100", "ARs"]
    # stats: 0 AP, 1 AP50, 2 AP75, 3 APs, 8 AR@100 -> use stat 8 as AR100
    vals = [stats[0], stats[1], stats[2], stats[3], stats[8], stats[9]]
    vals_n = np.clip(np.array(vals), 0, 1)
    angles = np.linspace(0, 2 * np.pi, len(labels_r), endpoint=False).tolist()
    vals_n = np.concatenate((vals_n, [vals_n[0]]))
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))
    ax.plot(angles, vals_n, "o-", linewidth=2)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_r)
    ax.set_title("Radar")
    fig.tight_layout()
    fig.savefig(plots / "05_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 06 P/R/F1 vs threshold
    sweep = thesis_data["threshold_sweep"]
    thrs = sorted([float(k) for k in sweep.keys()])
    pa = [sweep[str(t)]["all"]["precision"] for t in thrs]
    ra = [sweep[str(t)]["all"]["recall"] for t in thrs]
    fa = [sweep[str(t)]["all"]["f1"] for t in thrs]
    pc = [sweep[str(t)]["core"]["precision"] for t in thrs]
    rc = [sweep[str(t)]["core"]["recall"] for t in thrs]
    fc = [sweep[str(t)]["core"]["f1"] for t in thrs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(thrs, pa, label="P")
    axes[0].plot(thrs, ra, label="R")
    axes[0].plot(thrs, fa, label="F1")
    ba = thesis_data["best_all"]
    if ba.get("thr") is not None:
        axes[0].axvline(ba["thr"], color="gray", ls="--", label=f"best F1 thr={ba['thr']}")
    axes[0].set_title("All classes")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("threshold")
    axes[1].plot(thrs, pc, label="P")
    axes[1].plot(thrs, rc, label="R")
    axes[1].plot(thrs, fc, label="F1")
    bc = thesis_data["best_core"]
    if bc.get("thr") is not None:
        axes[1].axvline(bc["thr"], color="gray", ls="--", label=f"best F1 thr={bc['thr']}")
    axes[1].set_title("Core classes")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("threshold")
    fig.tight_layout()
    fig.savefig(plots / "06_pr_f1_vs_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 07 per-class P/R/F1 at best_all threshold
    thr_best = thesis_data["best_all"]["thr"]
    if thr_best is None:
        thr_best = 0.25
    snap = sweep.get(str(thr_best), sweep[sorted(sweep.keys())[len(sweep) // 2]])
    pc_d = snap["per_class"]
    fig, ax = plt.subplots(figsize=(12, 4))
    xs = np.arange(NUM_CLASSES)
    w = 0.25
    ps = [pc_d[str(i)]["precision"] for i in range(NUM_CLASSES)]
    rs = [pc_d[str(i)]["recall"] for i in range(NUM_CLASSES)]
    f1s = [pc_d[str(i)]["f1"] for i in range(NUM_CLASSES)]
    ax.bar(xs - w, ps, w, label="P")
    ax.bar(xs, rs, w, label="R")
    ax.bar(xs + w, f1s, w, label="F1")
    ax.set_xticks(xs)
    ax.set_xticklabels([VISDRONE_NAMES[i] for i in range(NUM_CLASSES)], rotation=30, ha="right")
    ax.legend()
    ax.set_title(f"Per-class P/R/F1 @ threshold={thr_best}")
    fig.tight_layout()
    fig.savefig(plots / "07_per_class_prf1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 08 TP vs FN
    snap2 = sweep.get(str(thr_best), snap)
    tps = [snap2["per_class"][str(i)]["tp"] for i in range(NUM_CLASSES)]
    fns = [snap2["per_class"][str(i)]["fn"] for i in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(NUM_CLASSES) - 0.2, tps, 0.4, label="TP", color="#1f4e79")
    ax.bar(np.arange(NUM_CLASSES) + 0.2, fns, 0.4, label="FN", color="#c00000")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(VISDRONE_NAMES, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "08_per_class_tp_fn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 09 thesis comparison (simplified bars)
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_k = ["AP50", "APs", "all_F1", "core_F1"]
    cur = [stats[1], stats[3], thesis_data["best_all"]["f1"], thesis_data["best_core"]["f1"]]
    base = [THESIS_BASE["ap50"], THESIS_BASE["aps"], THESIS_BASE["all_f1"], THESIS_BASE["core_f1"]]
    stretch = [THESIS_STRETCH["ap50"], THESIS_STRETCH["aps"], THESIS_STRETCH["all_f1"], THESIS_STRETCH["core_f1"]]
    x = np.arange(len(metrics_k))
    ax.bar(x, cur, color="#1f4e79", label="current")
    ax.plot(x, base, "k--", label="base target")
    ax.plot(x, stretch, color="#c5a000", ls="--", label="stretch target")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_k)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} vs thesis targets")
    fig.tight_layout()
    fig.savefig(plots / "09_thesis_target_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_report(
    model_name: str,
    stats: List[float],
    per_class_ap: Dict[str, Dict[str, float]],
    thesis_data: Dict[str, Any],
) -> str:
    lines = []
    lines.append("═" * 50)
    lines.append(f"  {model_name} 评测报告")
    lines.append("═" * 50)
    lines.append("  ── COCO 指标 ──")
    lines.append(
        f"  AP@0.5:        {stats[1]:.4f}    (基础≥{THESIS_BASE['ap50']} {'✅' if stats[1] >= THESIS_BASE['ap50'] else '❌'} / 争取≥{THESIS_STRETCH['ap50']} {'✅' if stats[1] >= THESIS_STRETCH['ap50'] else '❌'})"
    )
    lines.append(
        f"  APs:           {stats[3]:.4f}    (基础≥{THESIS_BASE['aps']} {'✅' if stats[3] >= THESIS_BASE['aps'] else '❌'} / 争取≥{THESIS_STRETCH['aps']} {'✅' if stats[3] >= THESIS_STRETCH['aps'] else '❌'})"
    )
    ba = thesis_data["best_all"]
    bc = thesis_data["best_core"]
    lines.append("  ── 任务书 P/R/F1（扫描阈值最优 F1）──")
    lines.append(
        f"  全类 F1:       {ba['f1']:.4f}    (基础≥{THESIS_BASE['all_f1']} {'✅' if ba['f1'] >= THESIS_BASE['all_f1'] else '❌'} / 争取≥{THESIS_STRETCH['all_f1']} {'✅' if ba['f1'] >= THESIS_STRETCH['all_f1'] else '❌'})"
    )
    lines.append(
        f"  核心类 F1:     {bc['f1']:.4f}    (基础≥{THESIS_BASE['core_f1']} {'✅' if bc['f1'] >= THESIS_BASE['core_f1'] else '❌'} / 争取≥{THESIS_STRETCH['core_f1']} {'✅' if bc['f1'] >= THESIS_STRETCH['core_f1'] else '❌'})"
    )
    lines.append("═" * 50)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_repo_on_path(args.dino_root)
    out_dir = resolve_output_dir(args)
    device = resolve_device(args.device)

    ckpt, model_state, cfg_from_ckpt = load_checkpoint_paths(args.resume)

    # 检测结果路径
    if args.det_json and Path(args.det_json).is_file():
        det_json_path = str(Path(args.det_json).resolve())
        import shutil

        shutil.copy(det_json_path, out_dir / "coco_instances_results.json")
    else:
        if not args.config_file:
            args.config_file = cfg_from_ckpt
        config_file = ensure_config_file(args, cfg_from_ckpt)
        det_json_path = run_inference_and_save(args, model_state, config_file, out_dir, device)

    thresholds = [round(0.05 + 0.05 * i, 2) for i in range(18)]  # 0.05 .. 0.90
    from pycocotools.coco import COCO

    coco_gt = COCO(args.val_ann_file)
    with open(det_json_path, "r", encoding="utf-8") as f:
        coco_results = json.load(f)

    stats, precision, _ = run_coco_eval(args.val_ann_file, det_json_path, args.max_dets)
    per_class_ap = per_class_ap_from_precision(precision)
    thesis_data = compute_prf1_at_thresholds(coco_results, coco_gt, thresholds)

    metrics_all = {
        "coco_stats": {f"stat_{i}": stats[i] for i in range(len(stats))},
        "per_class_ap50": {k: v["AP50"] for k, v in per_class_ap.items()},
        "per_class_aps": {k: v["APs"] for k, v in per_class_ap.items()},
        "per_class_ap": {k: v["AP"] for k, v in per_class_ap.items()},
        "thesis_pr_f1": {
            "best_all": thesis_data["best_all"],
            "best_core": thesis_data["best_core"],
            "per_class_at_best": {},  # 可选填充
        },
        "threshold_sweep": thesis_data["threshold_sweep"],
    }
    with open(out_dir / "metrics_all.json", "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)

    report = format_report(args.model_name, stats, per_class_ap, thesis_data)
    print(report)
    with open(out_dir / "terminal_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    plot_all(out_dir, stats, per_class_ap, thesis_data, args.model_name)
    print(f"\n完成。输出目录: {out_dir}")


if __name__ == "__main__":
    main()
