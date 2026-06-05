# MultiScaleDeformableAttention 重装（tools/reinstall_msda_for_current_gpu.py 与交互菜单共用）
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ops_dir() -> Path:
    return project_root() / "models" / "dino" / "ops"


def _path_on_colab_google_drive(p: Path) -> bool:
    """Colab 挂载 Drive 时路径含 /content/drive/；editable 安装常出现 pip 成功但无法 import 扩展。"""
    try:
        s = str(p.resolve()).replace("\\", "/")
    except OSError:
        s = str(p).replace("\\", "/")
    return "/content/drive/" in s


def detect_cuda_arch() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(0)
        return f"{p.major}.{p.minor}"
    except Exception:
        return None


def _rmtree_globs(ops: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for p in ops.glob(pattern):
            if p.is_dir():
                print(f"+ rm -rf {p}")
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                print(f"+ rm {p}")
                p.unlink(missing_ok=True)


def fixed_cuda_arch_for_memory_tier(memory_tier_idx: int) -> str | None:
    """
    交互菜单 ① 档与 MSDA「按档固定架构」对应。
    返回 None 表示应用「仅当前 GPU」（Colab G4 / T4 等非 A100/H100 固定档）。
    """
    mi = int(memory_tier_idx)
    # 与 training_interactive_menu ① 档顺序一致：0=A100-80 快速，1=T4，2=A100-40，3=G4，4=H100
    if mi == 0:
        return "8.0"
    if mi == 1:
        return None
    if mi == 2:
        return "8.0"
    if mi == 3:
        return None
    if mi == 4:
        return "9.0"
    return None


def run_msda_reinstall(
    *,
    single_arch: bool = False,
    fixed_arch: str | None = None,
) -> int:
    """
    在 models/dino/ops 下 pip uninstall + 清理 + pip install.
    - fixed_arch: 设置 TORCH_CUDA_ARCH_LIST（优先于 single_arch）
    - single_arch: 设置 DINO_MSDA_SINGLE_ARCH=1
    - 二者皆否: 使用 setup.py 默认多架构（含当前 GPU）
    - DINO_MSDA_NON_EDITABLE=1: 使用 `pip install .`（非 editable）
    - DINO_MSDA_NON_EDITABLE=0: 强制 `pip install -e .`，即使项目在 Drive 上
    - 未设置且项目位于 /content/drive/：自动使用 `pip install .`（与 Colab+Drive 兼容）
    返回 pip install 的退出码。
    """
    ops = ops_dir()
    if not ops.is_dir():
        print(f"[MSDA] ERROR: ops 目录不存在: {ops}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    if fixed_arch and fixed_arch.strip():
        env["TORCH_CUDA_ARCH_LIST"] = fixed_arch.strip()
        print(f"[MSDA] TORCH_CUDA_ARCH_LIST={env['TORCH_CUDA_ARCH_LIST']}")
    elif single_arch:
        env["DINO_MSDA_SINGLE_ARCH"] = "1"
        a = detect_cuda_arch()
        if a:
            print(f"[MSDA] DINO_MSDA_SINGLE_ARCH=1（当前 GPU arch {a}）")
        else:
            print("[MSDA] WARNING: 未检测到 CUDA，编译可能失败。", file=sys.stderr)
    else:
        a = detect_cuda_arch()
        if a:
            print(f"[MSDA] 使用 setup.py 默认多架构（含 {a}）")
        else:
            print("[MSDA] 未检测到 CUDA，将使用 setup.py 默认架构列表。")

    print(f"[MSDA] ops: {ops}")

    cmd_uninstall = [sys.executable, "-m", "pip", "uninstall", "-y", "MultiScaleDeformableAttention"]
    print("+", " ".join(cmd_uninstall))
    subprocess.run(cmd_uninstall, cwd=str(ops), env=env)

    _rmtree_globs(ops, ("build", "dist", "*.egg-info"))

    ne = os.environ.get("DINO_MSDA_NON_EDITABLE", "").strip().lower()
    if ne in ("1", "true", "yes", "y"):
        non_editable = True
    elif ne in ("0", "false", "no", "n"):
        non_editable = False
    else:
        non_editable = _path_on_colab_google_drive(ops) or _path_on_colab_google_drive(project_root())
        if non_editable:
            print(
                "[MSDA] 项目位于 Colab 的 Google Drive（/content/drive/），使用 pip install .（非 editable），"
                "避免 -e 安装后仍报 No module named 'MultiScaleDeformableAttention'。"
            )
    if non_editable:
        cmd_install = [sys.executable, "-m", "pip", "install", "-v", "."]
    else:
        cmd_install = [sys.executable, "-m", "pip", "install", "-v", "-e", "."]
    print("+", " ".join(cmd_install))
    r = subprocess.run(cmd_install, cwd=str(ops), env=env)
    return r.returncode


def run_msda_from_preset_dict(preset: dict) -> None:
    """
    根据 collect_visdrone_training_preset 返回的 msda_reinstall + memory_tier_idx 执行重装。
    """
    mode = preset.get("msda_reinstall", "skip")
    if mode == "skip":
        return
    mi = int(preset.get("memory_tier_idx", 0))

    if mode == "multi":
        rc = run_msda_reinstall()
    elif mode == "single":
        rc = run_msda_reinstall(single_arch=True)
    elif mode == "tier_fixed":
        fa = fixed_cuda_arch_for_memory_tier(mi)
        if fa:
            rc = run_msda_reinstall(fixed_arch=fa)
        else:
            rc = run_msda_reinstall(single_arch=True)
    else:
        return

    if rc != 0:
        print(
            f"\n[MSDA] pip install 退出码 {rc}。可稍后手动执行:\n"
            f"  python {project_root() / 'tools' / 'reinstall_msda_for_current_gpu.py'}\n",
            file=sys.stderr,
        )
    else:
        print("\n[MSDA] 重装完成，将继续启动训练。\n")
