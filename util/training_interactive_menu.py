# VisDrone thesis 交互式训练预设（供 main.py 与 tools/run_visdrone_training_menu 共用）
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _prompt_line(msg: str) -> str:
    try:
        return input(msg).strip().strip('"').strip("'")
    except EOFError:
        return ""


def _menu(title: str, items: list[tuple[str, str]]) -> int:
    print(f"\n{title}")
    for i, (key, desc) in enumerate(items, 1):
        print(f"  [{i}] {key}")
        if desc:
            for line in desc.split("\n"):
                print(f"      {line}")
    while True:
        raw = input("请选择序号 (回车默认 1): ").strip()
        if not raw:
            return 0
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(items):
                return n - 1
        print("无效输入，请重试。")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _yes_no(prompt: str, default: bool = True) -> bool:
    suf = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suf}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "是")


def _memory_tier(idx: int) -> dict[str, Any]:
    # ① 第一项：真 A100-80G 快速档（thesis_a100 / a100_quick；大 batch、关周期性 empty_cache）
    tiers = [
        {
            "name": "~80GB（A100-80G 快速，MSDA 用 sm_80）",
            "batch_full": 8,
            "lr_full": 2e-4,
            "lr_backbone_full": 2e-5,
            "batch_quick": 10,
            "lr_quick": 1.25e-4,
            "lr_backbone_quick": 1.25e-5,
            "cuda_empty": 0,
            "checkpoint_every_iters": 0,
            "crop_full": None,
            "num_workers_hint": 12,
        },
        {
            "name": "~16GB（T4 / 小显存，非 Colab 大显存 G4）",
            "batch_full": 2,
            "lr_full": 5e-5,
            "lr_backbone_full": 5e-6,
            "batch_quick": 2,
            "lr_quick": 2.5e-5,
            "lr_backbone_quick": 2.5e-6,
            "cuda_empty": 100,
            "checkpoint_every_iters": 500,
            "crop_full": "384,640",
            "num_workers_hint": 4,
        },
        {
            "name": "~40GB（A100-40G / L4 宽裕时）",
            "batch_full": 6,
            "lr_full": 1.5e-4,
            "lr_backbone_full": 1.5e-5,
            "batch_quick": 8,
            "lr_quick": 1e-4,
            "lr_backbone_quick": 1e-5,
            "cuda_empty": 0,
            "checkpoint_every_iters": 0,
            "crop_full": None,
            "num_workers_hint": 12,
        },
        {
            "name": "~80GB（Colab 大显存 G4，MSDA 按本机架构）",
            "batch_full": 8,
            "lr_full": 2e-4,
            "lr_backbone_full": 2e-5,
            "batch_quick": 10,
            "lr_quick": 1.25e-4,
            "lr_backbone_quick": 1.25e-5,
            "cuda_empty": 0,
            "checkpoint_every_iters": 0,
            "crop_full": None,
            "num_workers_hint": 12,
        },
        {
            "name": "~80GB（H100 80G，MSDA 用 sm_90）",
            "batch_full": 8,
            "lr_full": 2e-4,
            "lr_backbone_full": 2e-5,
            "batch_quick": 10,
            "lr_quick": 1.25e-4,
            "lr_backbone_quick": 1.25e-5,
            "cuda_empty": 0,
            "checkpoint_every_iters": 0,
            "crop_full": None,
            "num_workers_hint": 16,
        },
        {
            "name": "~60GB（H100 60G，速度档，MSDA 用 sm_90）",
            # 60G 下优先把每步更快：batch 不盲目拉满，避免偶发 OOM；lr 按 batch 近似线性缩放
            "batch_full": 6,
            "lr_full": 1.5e-4,
            "lr_backbone_full": 1.5e-5,
            "batch_quick": 8,
            "lr_quick": 1e-4,
            "lr_backbone_quick": 1e-5,
            "cuda_empty": 0,
            "checkpoint_every_iters": 0,
            "crop_full": None,
            "num_workers_hint": 16,
        },
    ]
    return tiers[idx]


def _epoch_plan(idx: int, memory_tier_idx: int) -> tuple[int, str, str]:
    plans = [
        (2, "快速栈（640 增强、GIoU、无通道注意力）", "DINO_4scale_visdrone_thesis_quick.py"),
        (6, "快速栈", "DINO_4scale_visdrone_thesis_quick.py"),
        (15, "论文完整栈（SIoU、通道注意力、大裁剪）", "DINO_4scale_visdrone_thesis.py"),
        (50, "论文完整栈", "DINO_4scale_visdrone_thesis.py"),
        (100, "论文完整栈（任务书长训）", "DINO_4scale_visdrone_thesis.py"),
        (100, "论文栈+EMA（毕业论文/PDF 提 AP）", "DINO_4scale_visdrone_graduation_max_ap.py"),
        (100, "轻量化+精度提升（5-scale 轻量化 + 蒸馏）", "DINO_5scale_visdrone_lightweight_100ep.py"),
    ]
    labels = [
        "冒烟（2 epoch，最快）",
        "短测（6 epoch）",
        "中短（15 epoch）",
        "中训（50 epoch）",
        "长训（100 epoch）",
        "长训+EMA（PDF 建议）",
        "轻量化+蒸馏（100 epoch）",
    ]
    e, desc, cfg = plans[idx]
    # ① 档 0：A100-80G 快速 → thesis_a100 / a100_quick（MSDA 固定 8.0）
    if memory_tier_idx == 0:
        if cfg == "DINO_4scale_visdrone_thesis_quick.py":
            cfg = "DINO_4scale_visdrone_thesis_a100_quick.py"
        elif cfg == "DINO_4scale_visdrone_thesis.py":
            cfg = "DINO_4scale_visdrone_thesis_a100.py"
        elif cfg == "DINO_4scale_visdrone_graduation_max_ap.py":
            cfg = "DINO_4scale_visdrone_graduation_max_ap_a100.py"
        desc = f"{desc} [A100-80G 快速]"
    # ① 档 1：真 16GB（T4）→ t4_16g 系列
    elif memory_tier_idx == 1:
        if cfg == "DINO_4scale_visdrone_thesis_quick.py":
            cfg = "DINO_4scale_visdrone_thesis_t4_16g_quick.py"
        elif cfg == "DINO_4scale_visdrone_thesis.py":
            cfg = "DINO_4scale_visdrone_thesis_t4_16g.py"
        elif cfg == "DINO_4scale_visdrone_graduation_max_ap.py":
            cfg = "DINO_4scale_visdrone_graduation_max_ap_t4_16g.py"
        desc = f"{desc} [T4/16G]"
    # ① 档 3：Colab 大显存 G4 → g4 系列（MSDA 勿用 8.0，按本机 Blackwell 等）
    elif memory_tier_idx == 3:
        if cfg == "DINO_4scale_visdrone_thesis_quick.py":
            cfg = "DINO_4scale_visdrone_thesis_g4_quick.py"
        elif cfg == "DINO_4scale_visdrone_thesis.py":
            cfg = "DINO_4scale_visdrone_thesis_g4.py"
        elif cfg == "DINO_4scale_visdrone_graduation_max_ap.py":
            cfg = "DINO_4scale_visdrone_graduation_max_ap_g4.py"
        desc = f"{desc} [Colab G4/80G]"
    # ① 档 4/5：H100 → h100 系列（MSDA 用 sm_90）
    elif memory_tier_idx in (4, 5):
        if cfg == "DINO_4scale_visdrone_thesis_quick.py":
            cfg = (
                "DINO_4scale_visdrone_thesis_h100_60g_quick.py"
                if memory_tier_idx == 5
                else "DINO_4scale_visdrone_thesis_h100_quick.py"
            )
        elif cfg == "DINO_4scale_visdrone_thesis.py":
            cfg = (
                "DINO_4scale_visdrone_thesis_h100_60g.py"
                if memory_tier_idx == 5
                else "DINO_4scale_visdrone_thesis_h100.py"
            )
        elif cfg == "DINO_4scale_visdrone_graduation_max_ap.py":
            cfg = (
                "DINO_4scale_visdrone_graduation_max_ap_h100_60g.py"
                if memory_tier_idx == 5
                else "DINO_4scale_visdrone_graduation_max_ap_h100.py"
            )
        desc = f"{desc} [H100]"
    return e, labels[idx] + " — " + desc, cfg


def collect_visdrone_training_preset(
    *,
    root: Optional[Path] = None,
    allow_skip_prompt: bool = True,
) -> Optional[dict[str, Any]]:
    """
    在 TTY 中收集预设。用户选择跳过时返回 None。
    allow_skip_prompt=False 时（独立启动器脚本）直接进入档位/epoch 菜单，不问「是否配置」。
    返回 dict: config_file, options (dict), num_workers, amp (bool)
    """
    if not sys.stdin.isatty():
        return None

    root = root or project_root()

    if allow_skip_prompt:
        print("\n======== ④ 训练预设（显存 / epoch / batch）========")
        raw = (_prompt_line("是否配置 VisDrone 训练预设？[Y/n]（回车=是）: ") or "y").lower()
        if raw in ("n", "no", "0", "否", "skip", "跳过"):
            print("已跳过，沿用当前 --config_file 与命令行 --options。\n")
            return None

    print("=" * 56)
    print(" VisDrone DINO — epoch / 显存档位 / batch")
    print("=" * 56)

    mi = _menu(
        "① 显存档位（建议 batch、学习率）",
        [
            ("~80GB — A100-80G 快速（推荐）", "config …_a100 / _a100_quick；大 batch；MSDA ③ 选「按档」→ 8.0"),
            ("~16GB — T4 / 真小显存", "config …_t4_16g；batch=2；裁剪 640"),
            ("~40GB — A100 40G 等", "batch 6/8 + lr 线性放大；cuda_empty=0 少同步；OOM 时 --options batch_size=4"),
            ("~80GB — Colab 大显存 G4", "config …_g4 / _g4_quick；MSDA 勿编 8.0；选「仅当前 GPU」或「多架构」"),
            ("~80GB — H100 80G", "config 默认论文栈；MSDA ③→ sm_90"),
            ("~60GB — H100 60G（速度档）", "batch=6（full）/8（quick）+ num_workers≈16；MSDA ③→ sm_90"),
        ],
    )
    tier = _memory_tier(mi)

    ei = _menu(
        "② 训练时长与配置栈",
        [
            ("2 epoch — 冒烟（quick）", ""),
            ("6 epoch — 短测（quick）", ""),
            ("15 epoch — 中短（完整论文栈）", ""),
            ("50 epoch — 中训（完整论文栈）", ""),
            ("100 epoch — 长训（完整论文栈）", ""),
            ("100 epoch — 长训+EMA（毕业论文/PDF 提 AP）", ""),
            ("100 epoch — 轻量化+蒸馏（lightweight_precision_boost_v3）", ""),
        ],
    )
    epochs, plan_label, cfg_name = _epoch_plan(ei, mi)
    config_file = str(root / "config" / "DINO" / cfg_name)
    full_stack = cfg_name in (
        "DINO_4scale_visdrone_thesis.py",
        "DINO_4scale_visdrone_thesis_g4.py",
        "DINO_4scale_visdrone_thesis_a100.py",
        "DINO_4scale_visdrone_thesis_t4_16g.py",
        "DINO_4scale_visdrone_graduation_max_ap.py",
        "DINO_4scale_visdrone_graduation_max_ap_g4.py",
        "DINO_4scale_visdrone_graduation_max_ap_a100.py",
        "DINO_4scale_visdrone_graduation_max_ap_t4_16g.py",
    )

    print(f"\n当前组合: [{tier['name']}] + [{plan_label}]")
    print(f"配置文件: config/DINO/{cfg_name}")

    if not _yes_no("是否按显存档位覆盖 batch_size / lr / lr_backbone？", default=True):
        tier_use = {
            **tier,
            "batch_full": None,
            "lr_full": None,
            "lr_backbone_full": None,
            "batch_quick": None,
            "lr_quick": None,
            "lr_backbone_quick": None,
        }
    else:
        tier_use = tier

    options: dict[str, Any] = {"epochs": epochs}
    extra_args: dict[str, Any] = {}

    if full_stack:
        if tier_use.get("batch_full") is not None:
            options["batch_size"] = tier_use["batch_full"]
            options["lr"] = tier_use["lr_full"]
            options["lr_backbone"] = tier_use["lr_backbone_full"]
        if tier_use.get("cuda_empty") is not None:
            options["cuda_empty_cache_every_iters"] = tier_use["cuda_empty"]
        if tier_use.get("checkpoint_every_iters") is not None:
            options["checkpoint_every_iters"] = tier_use["checkpoint_every_iters"]
        if tier.get("crop_full") and tier_use.get("batch_full") is not None:
            if _yes_no(
                f"缩小 RandomSizeCrop 上界为 [{tier['crop_full']}] 以省显存？",
                default=(mi == 1),
            ):
                a, b = tier["crop_full"].split(",")
                options["data_aug_scales2_crop"] = [int(a.strip()), int(b.strip())]
    else:
        if tier_use.get("batch_quick") is not None:
            options["batch_size"] = tier_use["batch_quick"]
            options["lr"] = tier_use["lr_quick"]
            options["lr_backbone"] = tier_use["lr_backbone_quick"]
        if tier_use.get("cuda_empty") is not None:
            options["cuda_empty_cache_every_iters"] = tier_use["cuda_empty"]
        if tier_use.get("checkpoint_every_iters") is not None:
            options["checkpoint_every_iters"] = tier_use["checkpoint_every_iters"]

    nw_default = tier["num_workers_hint"]
    nw_raw = _prompt_line(f"\nDataLoader num_workers [默认 {nw_default}]: ")
    num_workers = int(nw_raw) if nw_raw.isdigit() else nw_default

    use_amp = _yes_no("是否启用 --amp（混合精度，推荐）？", default=True)
    if cfg_name == "DINO_5scale_visdrone_lightweight_100ep.py":
        tckpt = _prompt_line("Teacher 权重路径（可选；回车=仅轻量化不蒸馏）: ").strip()
        if tckpt:
            extra_args["distill_teacher_ckpt"] = str(Path(tckpt).expanduser())
            extra_args["distill_weight"] = 1.0
            extra_args["distill_temperature"] = 4.0

    # ③ MSDA：与显存档位配套（main 交互结束后自动 pip 编译，无需再手敲命令）
    msda_mode = "skip"
    if os.environ.get("DINO_SKIP_MSDA_MENU", "").strip().lower() in ("1", "true", "yes", "y"):
        pass
    elif not _cuda_available():
        print(
            "\n[MSDA] 未检测到 CUDA，跳过 MSDA 编译（CPU 环境或未装 CUDA 版 PyTorch）。"
        )
    else:
        mj = _menu(
            "③ MSDA CUDA 扩展（与①显存档位配套；换 GPU / 首次训练前）",
            [
                ("跳过", "已编译过或稍后在命令行安装"),
                ("按当前 GPU 多架构重装（较慢）", "setup.py 默认；含 Blackwell 12.0 等"),
                ("仅当前 GPU（最快，Colab 推荐）", "DINO_MSDA_SINGLE_ARCH=1；换 GPU 需再执行本菜单"),
                (
                    "按①档固定架构重装",
                    "① 首项 A100-80→8.0；T4→本机；A100-40→8.0；Colab G4→本机(勿强编8.0)；H100→9.0",
                ),
            ],
        )
        msda_mode = ("skip", "multi", "single", "tier_fixed")[mj]

    return {
        "config_file": config_file,
        "options": options,
        "extra_args": extra_args,
        "num_workers": num_workers,
        "amp": use_amp,
        "memory_tier_idx": mi,
        "msda_reinstall": msda_mode,
    }


def preset_options_for_subprocess(preset: dict[str, Any]) -> list[str]:
    """将 preset['options'] 转为 main.py --options 的 argv 片段。"""
    argv: list[str] = []
    for k, v in preset["options"].items():
        if isinstance(v, list):
            argv.append(f"{k}={','.join(str(x) for x in v)}")
        elif isinstance(v, float):
            s = f"{v:.8f}".rstrip("0").rstrip(".")
            argv.append(f"{k}={s if s else '0'}")
        else:
            argv.append(f"{k}={v}")
    return argv


def merge_preset_into_args(args, preset: dict[str, Any]) -> None:
    """将 collect 结果写入 argparse Namespace；命令行已有 --options 的键优先覆盖菜单。"""
    args.config_file = preset["config_file"]
    args.num_workers = preset["num_workers"]
    if preset.get("amp"):
        args.amp = True

    menu_opts = dict(preset["options"])
    cli_opts = dict(args.options) if getattr(args, "options", None) else {}
    # 命令行 --options 优先于菜单
    args.options = {**menu_opts, **cli_opts}
    for k, v in dict(preset.get("extra_args", {})).items():
        setattr(args, k, v)


def prompt_visdrone_training_presets(args) -> None:
    """在 main 交互流程末尾调用；非 TTY 或用户跳过则不改写 args。"""
    preset = collect_visdrone_training_preset(allow_skip_prompt=True)
    if preset is None:
        return
    merge_preset_into_args(args, preset)
    print("已应用训练预设到本次运行。\n")
    from util.msda_reinstall import run_msda_from_preset_dict

    run_msda_from_preset_dict(preset)
