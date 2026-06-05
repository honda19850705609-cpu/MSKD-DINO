# 训练前检测 MultiScaleDeformableAttention CUDA 与当前 GPU 是否匹配，避免刷屏式 kernel 报错。
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch


def _ops_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "dino" / "ops"


def _clear_msda_import_cache() -> None:
    """pip 安装后清掉失败/旧模块缓存，便于再次 import。"""
    importlib.invalidate_caches()
    drop = [
        k
        for k in list(sys.modules.keys())
        if "MultiScaleDeformableAttention" in k or k.endswith("ms_deform_attn_func")
    ]
    for k in drop:
        del sys.modules[k]


def _auto_install_msda_if_missing(logger) -> bool:
    """
    未安装扩展时自动 pip 安装 ops（项目在 /content/drive/ 时默认 pip install .，避免 editable 无法 import）。
    DINO_NO_AUTO_MSDA_INSTALL=1 关闭；DINO_AUTO_MSDA_MULTI_ARCH=1 用多架构（慢，兼容换 GPU）。
    """
    if os.environ.get("DINO_NO_AUTO_MSDA_INSTALL", "").strip().lower() in ("1", "true", "yes", "y"):
        return False
    ops_s = str(_ops_dir())
    msg0 = (
        "[MSDA] 未检测到已安装的扩展，正在自动编译安装（首次或新环境）… "
        f"若失败请手动: cd {ops_s} && pip install -v -e ."
    )
    if logger is not None:
        logger.info(msg0)
    else:
        print(msg0)
    from util.msda_reinstall import fixed_cuda_arch_for_memory_tier, run_msda_reinstall

    multi = os.environ.get("DINO_AUTO_MSDA_MULTI_ARCH", "").strip().lower() in ("1", "true", "yes", "y")
    prof = os.environ.get("DINO_MSDA_INSTALL_PROFILE", "").strip().lower()
    tier_s = os.environ.get("DINO_MEMORY_TIER_IDX", "").strip()

    if prof in ("a100", "a100_80", "a100-80", "ampere"):
        rc = run_msda_reinstall(fixed_arch="8.0")
    elif prof in ("g4", "colab_g4", "blackwell", "b200"):
        rc = run_msda_reinstall() if multi else run_msda_reinstall(single_arch=True)
    elif prof in ("h100", "hopper"):
        rc = run_msda_reinstall(fixed_arch="9.0")
    elif tier_s.isdigit():
        fa = fixed_cuda_arch_for_memory_tier(int(tier_s))
        if fa:
            rc = run_msda_reinstall(fixed_arch=fa)
        else:
            rc = run_msda_reinstall() if multi else run_msda_reinstall(single_arch=True)
    else:
        rc = run_msda_reinstall() if multi else run_msda_reinstall(single_arch=True)
    _clear_msda_import_cache()
    if rc != 0:
        msg_e = f"[MSDA] 自动安装 pip 返回非零退出码 {rc}，请查看上方编译日志。"
        if logger is not None:
            logger.info(msg_e)
        else:
            print(msg_e)
        return False
    msg_ok = "[MSDA] 自动安装完成，重试加载扩展。"
    if logger is not None:
        logger.info(msg_ok)
    else:
        print(msg_ok)
    return True


def run_msda_cuda_preflight(device: torch.device, logger=None) -> None:
    """
    在 build_model 之前调用。失败时 SystemExit 并提示重装 ops（与 epoch/菜单配置无关）。
    设环境变量 DINO_SKIP_MSDA_PREFLIGHT=1 可跳过。
    """
    if os.environ.get("DINO_SKIP_MSDA_PREFLIGHT", "").strip() in ("1", "true", "yes", "y"):
        return
    if device.type != "cuda":
        return

    from util import misc as utils

    if utils.get_rank() != 0:
        return

    # 与训练时一致：通过 models.dino.ops 引用，避免 sys.path 插入 ops 导致与主流程导入分裂
    last_import_err: BaseException | None = None
    try:
        import MultiScaleDeformableAttention as _msda_mod  # noqa: F401
        from models.dino.ops.functions.ms_deform_attn_func import MSDeformAttnFunction
    except ImportError as e:
        last_import_err = e
        if _auto_install_msda_if_missing(logger):
            try:
                import MultiScaleDeformableAttention as _msda_mod  # noqa: F401
                from models.dino.ops.functions.ms_deform_attn_func import MSDeformAttnFunction
                last_import_err = None
            except ImportError as e2:
                last_import_err = e2
                # Colab+Drive 上偶发 pip 成功仍找不到模块：强制非 editable 再装一次
                on_drive = "/content/drive/" in str(_ops_dir()).replace("\\", "/")
                if on_drive and last_import_err is not None:
                    if logger is not None:
                        logger.info(
                            "[MSDA] 首次安装后仍无法 import，设置 DINO_MSDA_NON_EDITABLE=1 并重装…",
                        )
                    else:
                        print(
                            "[MSDA] 首次安装后仍无法 import，设置 DINO_MSDA_NON_EDITABLE=1 并重装…",
                        )
                    os.environ["DINO_MSDA_NON_EDITABLE"] = "1"
                    _clear_msda_import_cache()
                    if _auto_install_msda_if_missing(logger):
                        try:
                            import MultiScaleDeformableAttention as _msda_mod  # noqa: F401
                            from models.dino.ops.functions.ms_deform_attn_func import MSDeformAttnFunction
                            last_import_err = None
                        except ImportError as e3:
                            last_import_err = e3
        if last_import_err is not None:
            msg = (
                "\n[MSDA] 无法加载 MultiScaleDeformableAttention Python/CUDA 扩展。\n"
                f"请在项目根目录下执行:\n  cd {_ops_dir()}\n  pip install -v -e .\n"
                "或: python tools/reinstall_msda_for_current_gpu.py --profile a100|g4|h100\n"
                "项目在 /content/drive/ 时已默认用 pip install .；若仍失败可手动: "
                "cd models/dino/ops && pip uninstall -y MultiScaleDeformableAttention && pip install -v .\n"
                "若在本地仍想用 editable：export DINO_MSDA_NON_EDITABLE=0\n"
                "自动安装按 GPU 分流：DINO_MSDA_INSTALL_PROFILE=a100|g4|h100 或 DINO_MEMORY_TIER_IDX=0..4\n"
                "若不想自动安装：DINO_NO_AUTO_MSDA_INSTALL=1。\n"
                f"原始错误: {last_import_err}\n"
            )
            raise SystemExit(msg) from last_import_err

    ext_path = getattr(_msda_mod, "__file__", "(unknown)")
    if logger is not None:
        logger.info("MSDA extension: %s", ext_path)

    # 形状贴近 DINO：d_model=256, nheads=8 -> 每头 channels=32，走 col2im 的 case 32（与 test.py 里 D=2 不同）
    N, M, D = 2, 8, 32
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long, device=device)
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    S = int(sum([(H * W).item() for H, W in shapes]))

    dev_idx = device.index if device.index is not None else torch.cuda.current_device()

    value = (torch.rand(N, S, M, D, device=device, dtype=torch.float32) * 0.01).requires_grad_(True)
    sampling_locations = torch.rand(N, Lq, M, L, P, 2, device=device, dtype=torch.float32).requires_grad_(True)
    aw = torch.rand(N, Lq, M, L, P, device=device) + 1e-5
    attention_weights = (aw / aw.sum(-1, keepdim=True).sum(-2, keepdim=True)).requires_grad_(True)
    im2col_step = 2

    try:
        with torch.cuda.device(dev_idx):
            out = MSDeformAttnFunction.apply(
                value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step
            )
            out.backward(torch.ones_like(out))
            torch.cuda.synchronize(dev_idx)
    except Exception as e:
        hint_arch = ""
        try:
            p = torch.cuda.get_device_properties(dev_idx)
            hint_arch = f"当前 GPU: {p.name}, compute capability {p.major}.{p.minor}.\n"
        except Exception:
            pass
        msg = (
            "\n[MSDA] 可变形注意力 CUDA 扩展与当前 GPU 或驱动不匹配（与训练菜单 / 数据集无关；菜单只改 config 与超参）。\n"
            f"{hint_arch}"
            f"已加载扩展文件: {ext_path}\n"
            "典型原因: 在别的机器或别的 Colab GPU 类型上编译过 wheel，未在本会话 GPU 上重装。\n"
            "若日志仍出现旧版 printf 文案「error in ms_deformable_…」，说明运行中的 .so 不是当前源码编译的，请强制重装。\n\n"
            "推荐（自动按当前 GPU 架构重装，含 Blackwell 12.0 等）:\n"
            "  在项目根目录（与 main.py 同级）执行:\n"
            "    python tools/reinstall_msda_for_current_gpu.py\n"
            "  仅编当前卡、加快编译（换 GPU 需再跑一次）:\n"
            "    python tools/reinstall_msda_for_current_gpu.py --single-arch\n\n"
            "或手动在「当前已选 GPU」会话内:\n"
            f"  %cd {_ops_dir()}\n"
            "  !pip uninstall -y MultiScaleDeformableAttention\n"
            "  !rm -rf build *.egg-info\n"
            "  !pip install -v -e .\n"
            "（models/dino/ops/setup.py 会在未设置 TORCH_CUDA_ARCH_LIST 时自动包含当前 GPU 架构。）\n\n"
            "A100 勿在 Colab G4(Blackwell) 上复用 sm_80 的 .so：按机器重装。\n"
            "  python tools/reinstall_msda_for_current_gpu.py --profile a100|g4|h100\n"
            "  Colab G4 用 --profile g4（或 --single-arch）；真 A100 用 --profile a100\n"
            "  或: export TORCH_CUDA_ARCH_LIST='12.0' / '8.0' / '9.0'\n"
            f"\n原始异常: {e}\n"
        )
        raise SystemExit(msg) from e

    if logger is not None:
        logger.info("MSDA CUDA preflight: ok (im2col+col2im on cuda:%s, channels=%d).", dev_idx, D)
