# 三步快速启动：选 profile → 确认路径 → 汇总确认后开跑（替代多步问答）
# 由 main.py 在旧版交互菜单之前调用；返回 True 表示已写好 args，应跳过后续交互。
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _can_interact() -> bool:
    if not sys.stdin.isatty():
        return False
    if os.environ.get("RANK", "0") != "0":
        return False
    return True


def _prompt_line(msg: str) -> str:
    try:
        return input(msg).strip().strip('"').strip("'")
    except EOFError:
        return ""


def _yes_no(msg: str, default: bool = True) -> bool:
    suf = "[Y/n]" if default else "[y/N]"
    raw = input(f"{msg} {suf}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "是")


def _default_config_path() -> Path:
    return project_root() / "config" / "DINO" / "DINO_4scale.py"


def using_default_config_file(args: Any) -> bool:
    """仅当用户未显式指定非默认 -c 时启用 quick launch（避免覆盖用户意图）。"""
    try:
        return Path(args.config_file).expanduser().resolve() == _default_config_path().resolve()
    except Exception:
        return False


def dataset_paths_ready(args: Any) -> bool:
    """与 main.dataset_paths_ready 一致（避免 import main 循环依赖）。"""
    if (getattr(args, "coco_path", None) or "").strip():
        return True
    if (getattr(args, "dataset_json_root", None) or "").strip():
        return True
    if getattr(args, "train_ann_file", None) and getattr(args, "train_img_folder", None):
        return True
    return False


def apply_output_dir_from_env(args: Any) -> None:
    if (getattr(args, "output_dir", None) or "").strip():
        return
    for key in ("DINO_OUTPUT_DIR", "DINO_TRAIN_OUTPUT_DIR"):
        env = (os.environ.get(key) or "").strip()
        if env:
            args.output_dir = env
            return


def apply_pretrain_path_from_env(args: Any) -> None:
    if (getattr(args, "pretrain_model_path", None) or "").strip():
        return
    for key in ("DINO_PRETRAIN_MODEL_PATH", "DINO_PRETRAIN"):
        v = (os.environ.get(key) or "").strip()
        if v:
            args.pretrain_model_path = v
            return


def apply_user_dataset_paths(args: Any) -> None:
    from util.visdrone_dataset_pack import resolve_dataset_json_root_and_images

    resolve_dataset_json_root_and_images(args)


def _gpu_mem_gb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            b = int(torch.cuda.get_device_properties(0).total_memory)
            return b / (1024.0**3)
    except Exception:
        pass
    return None


def _gpu_desc() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            g = _gpu_mem_gb()
            if g is not None:
                return f"{name} | {g:.1f} GiB"
            return name
    except Exception:
        pass
    return "CPU 或无法检测 CUDA"


def batch_size_for_vram(gb: Optional[float]) -> int:
    """按显存粗分档覆盖 batch_size（4-scale VisDrone 栈经验值）。"""
    if gb is None:
        return 2
    if gb < 12:
        return 2
    if gb < 20:
        return 3
    if gb < 32:
        return 4
    if gb < 48:
        return 6
    return 8


def _merge_options(args: Any, profile_opts: dict[str, Any]) -> None:
    cli = dict(args.options) if getattr(args, "options", None) else {}
    args.options = {**profile_opts, **cli}


def detect_paths(args: Any) -> tuple[list[str], dict[str, Any]]:
    missing: list[str] = []
    out_dir = (getattr(args, "output_dir", None) or "").strip()
    if not out_dir:
        missing.append("output_dir")

    info: dict[str, Any] = {
        "output_dir": out_dir or "(未设置)",
        "dataset_mode": None,
        "coco_path": "",
        "dataset_json_root": "",
        "dataset_images": "",
        "pretrain_model_path": (getattr(args, "pretrain_model_path", None) or "").strip() or "(未设置，可选)",
    }

    if not dataset_paths_ready(args):
        missing.append("dataset")
    else:
        cp = (getattr(args, "coco_path", None) or "").strip()
        if cp:
            info["dataset_mode"] = "coco"
            info["coco_path"] = cp
        else:
            info["dataset_mode"] = "json"
            info["dataset_json_root"] = (getattr(args, "dataset_json_root", None) or "").strip()
            info["dataset_images"] = (getattr(args, "dataset_images", None) or "").strip()

    return missing, info


def _profiles() -> list[dict[str, Any]]:
    root = project_root()
    cfg = lambda name: str(root / "config" / "DINO" / name)
    return [
        {
            "key": "smoke",
            "label": "冒烟测试（2 epoch，验流程/管线）",
            "config_file": cfg("DINO_4scale_visdrone_thesis_quick.py"),
            "options": {"epochs": 2},
        },
        {
            "key": "baseline",
            "label": "基线训练（36 epoch，论文栈 thesis，标准超参）",
            "config_file": cfg("DINO_4scale_visdrone_thesis.py"),
            "options": {
                "epochs": 36,
                "multi_step_lr": True,
                "lr_drop_list": [30],
            },
        },
        {
            "key": "precision",
            "label": "精度冲线（36 epoch，cls=2.0 · focal_alpha=0.5 · EMA · 类均衡采样）【推荐】",
            "config_file": cfg("DINO_4scale_visdrone_graduation_max_ap.py"),
            "options": {
                "epochs": 36,
                "multi_step_lr": True,
                "lr_drop_list": [30],
                "cls_loss_coef": 2.0,
                "focal_alpha": 0.5,
                "use_ema": True,
                "ema_decay": 0.9997,
                "ema_epoch": 0,
                "train_class_balance_sampler": True,
                "train_balance_small_area_max": 1024,
                "train_balance_small_boost_scale": 0.55,
            },
        },
        {
            "key": "paper",
            "label": "论文冲线（50 epoch，阶段A 冲线配置 + 长训）",
            "config_file": cfg("DINO_4scale_visdrone_stageA_target_50ep.py"),
            "options": {},
        },
        {
            "key": "final_75ep",
            "label": "最终冲线（75 epoch，5-scale P2 + BiFPN + cls=3.0 · focal_alpha=0.6 · SIoU · 类别加权）",
            "config_file": cfg("DINO_5scale_visdrone_75ep_final.py"),
            "options": {},
        },
        {
            "key": "lightweight_100ep_distill",
            "label": "轻量化+精度提升（100 epoch，5-scale轻量化 + 蒸馏 + GPU自适应）【推荐】",
            "config_file": cfg("DINO_5scale_visdrone_lightweight_100ep.py"),
            "options": {},
            "needs_distill_prompt": True,
        },
        {
            "key": "eval_only",
            "label": "仅验证档（不训练，验证集 mAP；需 checkpoint，结构需与权重一致）",
            "config_file": cfg("DINO_4scale_visdrone_thesis.py"),
            "options": {},
            "eval_only": True,
        },
    ]


def _pick_profile() -> Optional[dict[str, Any]]:
    items = _profiles()
    print("\n======== ① 选择训练目标（Profile）========")
    for i, p in enumerate(items, 1):
        print(f"  [{i}] {p['label']}")
    print("  （回车默认 [6] 轻量化推荐档）")
    raw = _prompt_line(f"请选择序号 1-{len(items)}: ").strip()
    if not raw:
        idx = 5
    elif raw.isdigit():
        idx = int(raw) - 1
        if not (0 <= idx < len(items)):
            print("无效输入，已取消快速启动。")
            return None
    else:
        print("无效输入，已取消快速启动。")
        return None
    return items[idx]


def _collect_paths_to_args(args: Any, *, eval_only: bool = False) -> bool:
    """
    将路径写入 args（仅在本函数内失败时返回 False）。
    已齐全时展示并确认；用户否认则清空并重问缺失项。
    """
    missing, info = detect_paths(args)

    if not missing:
        print("\n======== ② 数据与输出路径（已从环境变量/命令行解析）========")
        print(f"  输出目录: {info['output_dir']}")
        if info["dataset_mode"] == "coco":
            print(f"  数据模式: 标准 COCO 目录\n  路径: {info['coco_path']}")
        elif info["dataset_mode"] == "json":
            print(
                f"  数据模式: COCO JSON + 图片\n  JSON 根: {info['dataset_json_root']}\n  图片: {info['dataset_images']}"
            )
        print(f"  预训练权重: {info['pretrain_model_path']}")
        if not _yes_no("\n以上路径是否正确？", default=True):
            args.output_dir = ""
            args.coco_path = ""
            args.dataset_json_root = ""
            args.dataset_images = ""
            missing = ["output_dir", "dataset"]

    if "output_dir" in missing:
        o = _prompt_line("输出目录（checkpoint / 日志，必填）: ")
        if not o:
            print("未设置输出目录，已取消。")
            return False
        args.output_dir = str(Path(o).expanduser())

    if "dataset" in missing or not dataset_paths_ready(args):
        print("\n数据（二选一）:\n  1 = 标准 COCO 根目录\n  2 = COCO JSON 根目录 + 图片文件夹")
        ch = (_prompt_line("请选择 [1/2]，默认 2: ") or "2").strip()
        if ch == "1":
            p = _prompt_line("COCO 根目录: ")
            if not p:
                print("未输入路径，已取消。")
                return False
            args.coco_path = str(Path(p).expanduser().resolve())
            args.dataset_json_root = ""
            args.dataset_images = ""
        else:
            jr = _prompt_line("JSON 根目录（含 annotations/instances_*.json）: ")
            im = _prompt_line("图片文件夹: ")
            if not jr or not im:
                print("JSON 与图片路径均需填写，已取消。")
                return False
            args.dataset_json_root = str(Path(jr).expanduser().resolve())
            args.dataset_images = str(Path(im).expanduser().resolve())
            args.coco_path = ""

    if eval_only:
        ck = _prompt_line("验证用 checkpoint .pth（必填；与模型结构一致）: ").strip()
        if not ck:
            print("未提供权重，已取消。")
            return False
        args.resume = str(Path(ck).expanduser())
    else:
        pr = _prompt_line("COCO 预训练 .pth（可选，回车跳过）: ").strip()
        if pr:
            args.pretrain_model_path = str(Path(pr).expanduser())

    return True


def quick_launch(args: Any) -> bool:
    """
    三步向导。返回 True 表示已完成配置，main 应跳过后续旧版交互。
    """
    if not _can_interact():
        return False
    if not using_default_config_file(args):
        return False
    if os.environ.get("DINO_NO_QUICK_LAUNCH", "").strip().lower() in ("1", "true", "yes", "y"):
        return False

    prof = _pick_profile()
    if prof is None:
        return False

    eval_only = bool(prof.get("eval_only"))
    if eval_only:
        args.eval = True
    if prof.get("needs_distill_prompt"):
        tckpt = _prompt_line("Teacher 权重 .pth（建议 checkpoint_best_ema.pth；回车=仅轻量化不蒸馏）: ").strip()
        if tckpt:
            args.distill_teacher_ckpt = str(Path(tckpt).expanduser())
            args.distill_weight = 1.0
            args.distill_temperature = 4.0

    gb = _gpu_mem_gb()
    bs = batch_size_for_vram(gb)
    profile_opts = dict(prof["options"])
    if not eval_only:
        profile_opts["batch_size"] = bs

    apply_user_dataset_paths(args)
    apply_output_dir_from_env(args)
    if not eval_only:
        apply_pretrain_path_from_env(args)

    if not _collect_paths_to_args(args, eval_only=eval_only):
        return False

    apply_user_dataset_paths(args)

    import multiprocessing

    nw = min(8, int(multiprocessing.cpu_count() or 8))
    args.num_workers = nw
    args.amp = True

    opt_show = {**profile_opts}
    print("\n======== ③ 确认启动 =========")
    print(f"  Profile: {prof['label']}")
    print(f"  配置: {Path(prof['config_file']).name}")
    print(f"  GPU: {_gpu_desc()}")
    if eval_only:
        print("  模式: 仅验证（--eval），不训练")
        print("  batch_size: （沿用配置；验证通常为 1）")
    else:
        print(f"  batch_size（按显存自动）: {bs}")
    print(f"  num_workers: {nw}")
    print(f"  AMP: True")
    if not eval_only:
        print(f"  epochs: {opt_show.get('epochs', '(见配置文件)')}")
    if "cls_loss_coef" in opt_show:
        print(f"  cls_loss_coef: {opt_show.get('cls_loss_coef')}")
    if "focal_alpha" in opt_show:
        print(f"  focal_alpha: {opt_show.get('focal_alpha')}")
    if opt_show.get("use_ema"):
        print(f"  EMA: use_ema=True, ema_decay={opt_show.get('ema_decay', '')}")
    print(f"  output_dir: {args.output_dir}")
    if (args.coco_path or "").strip():
        print(f"  数据: COCO — {args.coco_path}")
    else:
        print(f"  数据: JSON — {args.dataset_json_root}")
        print(f"        图片 — {args.dataset_images}")
    if eval_only:
        print(f"  resume: {(getattr(args, 'resume', None) or '').strip()}")
    else:
        pre = (getattr(args, "pretrain_model_path", None) or "").strip()
        print(f"  pretrain: {pre or '(无)'}")
        if getattr(args, "distill_teacher_ckpt", ""):
            print(f"  distill_teacher_ckpt: {args.distill_teacher_ckpt}")
            print(f"  distill_weight: {getattr(args, 'distill_weight', 1.0)}")
            print(f"  distill_temperature: {getattr(args, 'distill_temperature', 4.0)}")

    if not _yes_no("\n确认开始验证？" if eval_only else "\n确认开始训练？", default=True):
        print("已取消快速启动。\n")
        return False

    args.config_file = prof["config_file"]
    _merge_options(args, profile_opts)

    print("已应用 Quick Launch 配置。\n")
    return True
