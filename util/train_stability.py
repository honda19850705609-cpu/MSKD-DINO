"""降低训练峰值负载，减轻笔记本黑屏/重启概率（供电与散热仍须自行保证）。"""
from __future__ import annotations

import os


def apply_laptop_stable_training(args, logger=None) -> None:
    """
    由 config 中 laptop_stable_training=True 或环境变量 DINO_LAPTOP_STABLE=1 触发。
    - 限制 DataLoader num_workers
    - 关闭 cudnn.benchmark（略降速，部分机型更稳）
    - 限制 PyTorch 本进程 CPU 计算线程，避免与 DataLoader 子进程抢满 CPU
    """
    env_on = (os.environ.get('DINO_LAPTOP_STABLE') or '').strip().lower() in ('1', 'true', 'yes', 'on')
    if not env_on and not getattr(args, 'laptop_stable_training', False):
        return

    import torch

    torch.backends.cudnn.benchmark = False

    cap = int(getattr(args, 'laptop_stable_num_workers_cap', 4) or 4)
    nw = int(getattr(args, 'num_workers', 10) or 0)
    if nw > cap:
        if logger:
            logger.info('laptop_stable_training: num_workers %s -> %s', nw, cap)
        args.num_workers = cap

    try:
        import multiprocessing
        cpu = multiprocessing.cpu_count() or 8
        nt = max(1, min(4, max(1, cpu // 4)))
        torch.set_num_threads(nt)
        if logger:
            logger.info(
                'laptop_stable_training: cudnn.benchmark=False, torch.set_num_threads(%s)',
                nt,
            )
    except Exception:
        if logger:
            logger.info('laptop_stable_training: cudnn.benchmark=False (set_num_threads skipped)')
