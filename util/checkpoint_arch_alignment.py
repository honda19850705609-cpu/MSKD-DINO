"""根据 checkpoint 的 state_dict 键名，补全与训练时不一致的 args.options（如 add_channel_attention）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from util.misc import find_checkpoint_for_resume, torch_load_trusted


def resolve_checkpoint_path_for_peek(args: Any) -> Optional[str]:
    """与 main 一致：--resume → --pretrain_model_path → output_dir 下自动查找。"""
    p = (getattr(args, 'resume', None) or '').strip()
    if p and Path(p).is_file():
        return str(Path(p).resolve())
    pre = (getattr(args, 'pretrain_model_path', None) or '').strip()
    if pre:
        pr = Path(pre).expanduser()
        if pr.is_file():
            return str(pr.resolve())
    out = (getattr(args, 'output_dir', None) or '').strip()
    if out and not getattr(args, 'no_auto_resume', False):
        ck = find_checkpoint_for_resume(out)
        if ck:
            return ck
    return None


def maybe_align_options_from_checkpoint_arch(args: Any) -> None:
    """
    若权重中含 transformer.encoder.*.activ_channel（VisDrone thesis 通道注意力），
    而当前配置未打开 add_channel_attention，则写入 options，避免 load_state_dict 报 Unexpected key。
    """
    if getattr(args, 'no_auto_ckpt_arch', False):
        return
    ckpt_path = resolve_checkpoint_path_for_peek(args)
    if not ckpt_path:
        return
    try:
        ckpt = torch_load_trusted(ckpt_path, map_location='cpu')
        sd = ckpt.get('model', ckpt)
        if not isinstance(sd, dict):
            return
        keys = tuple(sd.keys())
    except Exception:
        return
    has_ch = any('activ_channel' in k for k in keys)
    if not has_ch:
        return
    opt = getattr(args, 'options', None)
    if opt is not None and 'add_channel_attention' in opt:
        return
    if args.options is None:
        args.options = {}
    args.options['add_channel_attention'] = True
    print(
        f'[checkpoint] 从「{Path(ckpt_path).name}」检测到编码器通道注意力(activ_channel)，'
        '已设置 add_channel_attention=True。若需关闭自动对齐请加 --no_auto_ckpt_arch。'
    )
