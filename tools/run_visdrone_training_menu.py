#!/usr/bin/env python3
"""
独立启动器：交互选择显存档位 / epoch / batch 后子进程执行 main.py。
（与在项目根目录直接 python main.py 后出现的菜单逻辑一致，见 util/training_interactive_menu.py）

用法:
  cd <项目根目录>
  python tools/run_visdrone_training_menu.py

  仅打印命令不执行:
  python tools/run_visdrone_training_menu.py --dry-run

  跟在 -- 后面的参数会原样传给 main.py:
  python tools/run_visdrone_training_menu.py -- --output_dir D:/out --dataset_json_root ... --amp
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.msda_reinstall import run_msda_from_preset_dict
from util.training_interactive_menu import (
    collect_visdrone_training_preset,
    preset_options_for_subprocess,
    project_root,
)


def _split_argv() -> tuple[list[str], list[str]]:
    if "--" in sys.argv:
        i = sys.argv.index("--")
        return sys.argv[1:i], sys.argv[i + 1 :]
    return sys.argv[1:], []


def main() -> None:
    launcher_argv, extra_from_dash = _split_argv()
    ap = argparse.ArgumentParser(description="VisDrone thesis 训练交互菜单（子进程 main）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令，不运行")
    la = ap.parse_args(launcher_argv)
    extra_main = list(extra_from_dash)

    os.chdir(project_root())

    preset = collect_visdrone_training_preset(root=project_root(), allow_skip_prompt=False)
    if preset is None:
        raise SystemExit("无法收集预设（需要交互式终端）。")

    run_msda_from_preset_dict(preset)

    opt_argv = preset_options_for_subprocess(preset)
    cmd: list[str] = [
        sys.executable,
        str(project_root() / "main.py"),
        "-c",
        preset["config_file"],
        "--num_workers",
        str(preset["num_workers"]),
    ]
    if opt_argv:
        cmd.append("--options")
        cmd.extend(opt_argv)
    if preset.get("amp"):
        cmd.append("--amp")
    cmd.extend(extra_main)

    print("\n" + "-" * 56)
    print("将执行:")
    try:
        print(" ", shlex.join(cmd))
    except AttributeError:
        print(" ", " ".join(shlex.quote(c) for c in cmd))
    print("-" * 56)

    if la.dry_run:
        print("[--dry-run] 未启动训练。")
        return

    if not extra_main or not any(
        x.startswith("--output_dir")
        or x.startswith("--output=")
        or x == "--output"
        or x.startswith("--dataset_json_root")
        or x.startswith("--coco_path")
        or x.startswith("--thesis_visdrone")
        for x in extra_main
    ):
        print(
            "\n提示: 未检测到输出目录或数据路径。请追加例如:\n"
            "  --output_dir <保存目录> --dataset_json_root <COCO根> --dataset_images <图片文件夹>\n"
        )
        try:
            ok = input("仍要继续执行？ [y/N]: ").strip().lower() in ("y", "yes", "1", "是")
        except EOFError:
            ok = False
        if not ok:
            print("已取消。")
            return

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    ret = subprocess.run(cmd, cwd=str(project_root()), env=env)
    raise SystemExit(ret.returncode)


if __name__ == "__main__":
    main()
