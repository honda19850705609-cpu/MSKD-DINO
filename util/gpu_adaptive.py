"""
零精度损失的 GPU 自适应系统。

只调整：batch_size（配合梯度累积）和 num_workers。
绝不修改：模型架构、增强参数、损失权重、学习率等任何影响精度的参数。

原理：
  配置文件写 batch_size=4，GPU 只能跑 batch_size=2 时：
  → 实际 batch_size 设为 2
  → 梯度累积步数设为 2
  → 每 2 个 mini-batch 做一次 optimizer.step
  → 等效梯度 = 4 张图的平均梯度，和原始配置完全一致
  → LR 不需要调整，训练结果数学上等价
"""

from __future__ import annotations

import gc
import os
from typing import Any, Callable, Tuple

import torch


def detect_gpu_info() -> dict:
    """检测 GPU 基本信息"""
    if not torch.cuda.is_available():
        return {"available": False, "name": "CPU", "total_gb": 0.0, "device_index": None}

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total_gb = float(props.total_memory) / (1024.0**3)
    return {
        "available": True,
        "name": props.name,
        "total_gb": round(total_gb, 1),
        "device_index": int(idx),
    }


def probe_max_batch_size(
    *,
    build_model_fn: Callable[[Any], Tuple[Any, Any, Any]],
    args: Any,
    device: torch.device,
    target_batch: int,
    min_batch: int = 1,
    logger=None,
) -> tuple[int, int]:
    """
    用真实的前向+反向探测当前 GPU 能跑的最大 batch_size。

    从 target_batch 开始尝试，OOM 就减 1，直到成功或降到 min_batch。
    探测完成后清理显存，不留残留。

    返回: (actual_batch_size, gradient_accumulation_steps)
    """

    def log(msg: str) -> None:
        if logger:
            logger.info(f"[GPU探测] {msg}")
        print(f"[GPU探测] {msg}")

    gpu = detect_gpu_info()
    if gpu["available"]:
        log(f"GPU: {gpu['name']} ({gpu['total_gb']} GB)")
    else:
        log("未检测到 GPU，batch_size=1")
        return 1, max(1, int(target_batch))

    log(f"目标 batch_size={target_batch}，开始探测实际可用值...")

    for try_batch in range(int(target_batch), int(min_batch) - 1, -1):
        success = _try_forward_backward(
            build_model_fn=build_model_fn,
            args=args,
            device=device,
            batch_size=int(try_batch),
            logger=logger,
        )
        if success:
            accum = max(1, int(target_batch) // int(try_batch))
            effective = int(try_batch) * int(accum)
            if int(try_batch) == int(target_batch):
                log(f"batch_size={try_batch} 可用，无需调整")
            else:
                log(f"batch_size={try_batch} 可用")
                log(f"梯度累积步数={accum}，等效 batch={effective}")
                if effective != int(target_batch):
                    log(
                        f"注意：等效 batch={effective} ≠ 原始 {target_batch}"
                        f"（{target_batch} 不能被 {try_batch} 整除）"
                    )
            return int(try_batch), int(accum)

    log("⚠ 即使 batch_size=1 也 OOM，请检查模型配置或换更大显存的 GPU")
    return 1, max(1, int(target_batch))


def _try_forward_backward(
    *,
    build_model_fn,
    args,
    device: torch.device,
    batch_size: int,
    logger=None,
) -> bool:
    """
    用最大尺寸的 dummy tensor 做前向+反向探测。
    确保探测的是最坏情况（最大图），实际训练不可能比这更大。
    成功返回 True，OOM 返回 False。
    无论成功失败都清理显存。
    """

    def log(msg: str) -> None:
        if logger:
            logger.info(f"[GPU探测] {msg}")
        print(f"[GPU探测] {msg}")

    max_size = int(getattr(args, "data_aug_max_size", 1333) or 1333)
    # ★ 绝对最坏情况：max_size × max_size 正方形
    h = max(32, int(max_size))
    w = max(32, int(max_size))
    # ★ VisDrone 密集场景：提高 dummy GT 数（更保守）
    n_gt = int(getattr(args, "gpu_adaptive_dummy_n_gt", 80) or 80)
    n_gt = max(1, n_gt)
    safety_reserve_ratio = float(getattr(args, "gpu_adaptive_safety_reserve_ratio", 0.10) or 0.10)
    safety_reserve_ratio = min(max(safety_reserve_ratio, 0.05), 0.25)
    usable_ratio = 1.0 - safety_reserve_ratio
    has_distill = bool(getattr(args, "distill_teacher_ckpt", None))
    # teacher(frozen, no grad) + runtime buffer，避免探测通过后加载蒸馏再 OOM
    distill_overhead_gb = float(getattr(args, "gpu_adaptive_distill_overhead_gb", 4.0) or 4.0) if has_distill else 0.0

    log(
        f"  尝试 batch_size={batch_size} (绝对最坏 max={w}x{h}，n_gt={n_gt}，重复 3 次取峰值，"
        f"预留显存={safety_reserve_ratio:.0%}{'，含蒸馏额外预留' if has_distill else ''})..."
    )

    try:
        if device.type != "cuda":
            # CPU 情况下不做显存比例判定
            log(f"  batch_size={batch_size}：当前为 CPU/非CUDA，跳过显存探测，直接通过。")
            return True

        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_idx)
        total_gb = float(props.total_memory) / (1024.0**3)

        peak_gb_max = 0.0
        # ★ 每个 batch_size 探测 3 次，取峰值最大的一次（更保守）
        for trial in range(3):
            # 不同随机种子（但不修改全局 seed）
            g = torch.Generator(device=device)
            g.manual_seed(12345 + trial * 1000 + int(batch_size))

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model, criterion, postprocessors = build_model_fn(args)
            model.to(device)
            model.train()
            criterion.train()

            from util.misc import NestedTensor

            dummy_tensors = torch.randn(batch_size, 3, h, w, device=device, generator=g)
            dummy_mask = torch.zeros(batch_size, h, w, dtype=torch.bool, device=device)
            samples = NestedTensor(dummy_tensors, dummy_mask)

            # dummy targets (cxcywh normalized in [0,1])
            nc = int(getattr(args, "num_classes", 10) or 10)
            targets = []
            for _ in range(batch_size):
                targets.append(
                    {
                        "labels": torch.randint(0, nc, (n_gt,), device=device, generator=g),
                        "boxes": torch.rand(n_gt, 4, device=device, generator=g).clamp(0.01, 0.99),
                        "orig_size": torch.tensor([h, w], device=device),
                        "size": torch.tensor([h, w], device=device),
                    }
                )

            if getattr(args, "amp", False):
                with torch.cuda.amp.autocast():
                    outputs = model(samples, targets) if getattr(args, "use_dn", False) else model(samples)
                    loss_dict = criterion(outputs, targets)
            else:
                outputs = model(samples, targets) if getattr(args, "use_dn", False) else model(samples)
                loss_dict = criterion(outputs, targets)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses.backward()

            peak_gb = float(torch.cuda.max_memory_allocated()) / (1024.0**3)
            peak_gb_max = max(peak_gb_max, peak_gb)
            log(f"    trial {trial+1}/3 peak: {peak_gb:.1f} GB (max so far: {peak_gb_max:.1f} GB)")

            # 清理本次 trial
            del model, criterion, postprocessors, outputs, loss_dict, losses
            del samples, dummy_tensors, dummy_mask, targets
            gc.collect()
            torch.cuda.empty_cache()

        effective_peak_gb = peak_gb_max + distill_overhead_gb
        usage_ratio = effective_peak_gb / max(total_gb, 1e-6)
        if usage_ratio > usable_ratio:
            log(
                f"  batch_size={batch_size} 不安全：有效峰值 {effective_peak_gb:.1f}GB "
                f"(实测 {peak_gb_max:.1f}GB + 蒸馏预留 {distill_overhead_gb:.1f}GB) / {total_gb:.1f}GB"
                f" = {usage_ratio:.0%} > {usable_ratio:.0%}（需预留 {safety_reserve_ratio:.0%} 余量）"
            )
            return False

        log(
            f"  batch_size={batch_size} 通过：有效峰值 {effective_peak_gb:.1f}GB "
            f"(实测 {peak_gb_max:.1f}GB + 蒸馏预留 {distill_overhead_gb:.1f}GB) / {total_gb:.1f}GB"
            f" = {usage_ratio:.0%}（<={usable_ratio:.0%} 安全）"
        )
        return True

    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg:
            log(f"  batch_size={batch_size} OOM")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return False
        raise
    except Exception:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise


def auto_adapt_batch(*, args, build_model_fn, device: torch.device, logger=None) -> None:
    """
    主入口。在 main.py 中模型正式构建之前调用。

    只修改 args 中的三个字段：
    - args.batch_size（可能降低）
    - args.gradient_accumulation_steps（新增，补偿降低的 batch）
    - args.num_workers（根据 CPU 核心数调整）

    绝不修改其他任何字段。
    """

    def log(msg: str) -> None:
        if logger:
            logger.info(msg)
        print(msg)

    target_batch = int(getattr(args, "batch_size", 4) or 4)

    log("")
    log("=" * 55)
    log("  GPU 自适应（零精度损失）")
    log("=" * 55)

    actual_batch, accum_steps = probe_max_batch_size(
        build_model_fn=build_model_fn,
        args=args,
        device=device,
        target_batch=target_batch,
        logger=logger,
    )

    args.batch_size = int(actual_batch)
    args.gradient_accumulation_steps = int(accum_steps)

    cpu_count = int(os.cpu_count() or 4)
    current_workers = int(getattr(args, "num_workers", 4) or 4)
    recommended = min(current_workers, cpu_count, int(actual_batch) * 4, 12)
    args.num_workers = max(2, int(recommended))

    log("")
    log("  最终配置:")
    log(f"    batch_size:        {actual_batch}  (配置文件: {target_batch})")
    log(f"    梯度累积:          {accum_steps}  步")
    log(f"    等效 batch:        {actual_batch * accum_steps}")
    log(f"    num_workers:       {args.num_workers}")
    log("    其他参数:          全部保持配置文件原值（不修改）")
    log("=" * 55)
    log("")

