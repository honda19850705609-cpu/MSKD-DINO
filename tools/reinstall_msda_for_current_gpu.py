#!/usr/bin/env python3
"""
在「当前会话已选中的 GPU」上重装 MultiScaleDeformableAttention。

用法见 util/msda_reinstall.run_msda_reinstall；main 交互菜单③ 会调用同一逻辑。

A100 与 Colab 大显存 G4（常为 Blackwell 等）架构不同，请分开编译：
  python tools/reinstall_msda_for_current_gpu.py --profile a100
  python tools/reinstall_msda_for_current_gpu.py --profile g4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.msda_reinstall import run_msda_reinstall  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Reinstall MSDA ops for the current CUDA device.")
    parser.add_argument(
        "--profile",
        type=str,
        default="",
        metavar="a100|g4|h100",
        help="a100→sm_80；g4→仅当前 GPU（Colab G4/Blackwell）；h100→sm_90",
    )
    parser.add_argument(
        "--single-arch",
        action="store_true",
        help="Set DINO_MSDA_SINGLE_ARCH=1 so setup.py only compiles for GPU 0 (fastest).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="",
        help="Force TORCH_CUDA_ARCH_LIST, e.g. 12.0 for Blackwell.",
    )
    args = parser.parse_args()
    if args.arch.strip():
        return run_msda_reinstall(fixed_arch=args.arch.strip())
    p = (args.profile or "").strip().lower()
    if p and p not in ("a100", "g4", "h100"):
        print("ERROR: --profile must be a100, g4, or h100", file=sys.stderr)
        return 2
    if p == "a100":
        return run_msda_reinstall(fixed_arch="8.0")
    if p == "g4":
        return run_msda_reinstall(single_arch=True)
    if p == "h100":
        return run_msda_reinstall(fixed_arch="9.0")
    if args.single_arch:
        return run_msda_reinstall(single_arch=True)
    return run_msda_reinstall()


if __name__ == "__main__":
    raise SystemExit(main())
