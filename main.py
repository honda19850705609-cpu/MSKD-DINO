# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import gc
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils
from util.train_stability import apply_laptop_stable_training
from util.lightweight_boost import setup_distillation, log_gpu, efficiency_report

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test

_DEFAULT_CONFIG_FILE = str(Path(__file__).resolve().parent / 'config' / 'DINO' / 'DINO_4scale.py')


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument(
        '--config_file', '-c', type=str, default=_DEFAULT_CONFIG_FILE,
        help='Path to config .py file. Default: config/DINO/DINO_4scale.py (next to main.py).',
    )
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument(
        '--coco_path', '--dataset', '--data', '--dataset_dir', '--dataset_path', '--data_root',
        type=str, default='',
        dest='coco_path',
        metavar='PATH',
        help='标准 COCO 目录根路径（含 train2017/、val2017/、annotations/）。与 --dataset_json_root 二选一。'
             '常用别名: --dataset, --data。',
    )
    parser.add_argument(
        '--dataset_json_root', type=str, default='',
        metavar='PATH',
        help='COCO 标注根目录：须含 annotations/instances_train2017.json 与 instances_val2017.json。'
             '若为自包含数据包（另有 images/、可选 dataset_manifest.json），未指定 --dataset_images 时'
             '默认使用包内 images/ 作为 train 与 val 图像目录。'
             '环境变量: DINO_DATASET_JSON_ROOT 或 DINO_DATASET_PACK_ROOT（同义）。'
             '设置后清空 --coco_path。',
    )
    parser.add_argument(
        '--dataset_images', type=str, default='',
        metavar='PATH',
        help='与 --dataset_json_root 配套：图片所在文件夹（与 JSON 里 file_name 一致）。'
             '未单独指定 --train_img_folder 时同时作为训练/验证图像目录。环境变量: DINO_DATASET_IMAGES。',
    )
    parser.add_argument('--train_img_folder', type=str, default=None,
                        help='Training images folder (pair with --train_ann_file; overrides COCO layout).')
    parser.add_argument('--train_ann_file', type=str, default=None,
                        help='Training COCO instances JSON path (pair with --train_img_folder).')
    parser.add_argument('--val_img_folder', type=str, default=None,
                        help='Val images folder (default: same as --train_img_folder).')
    parser.add_argument('--val_ann_file', type=str, default=None,
                        help='Val COCO instances JSON (default: same as --train_ann_file).')
    parser.add_argument(
        '--thesis_visdrone_50', action='store_true',
        help='使用 config/DINO/DINO_4scale_visdrone_thesis.py（10 类）及 util/thesis_spec.py 中的默认数据路径；'
             '若尚无 COCO JSON，启动时会从 *_cleaned.txt 自动生成。可用 THESIS_COCO_ROOT、THESIS_IMAGES_DIR 覆盖。',
    )
    parser.add_argument(
        '--no_auto_visdrone_coco', action='store_true',
        help='禁止自动从 *_cleaned.txt 生成 COCO JSON（缺 instances_*.json 时直接报错）。',
    )
    parser.add_argument(
        '--thesis_taskbook_pdf', type=str, default='',
        help='可选：任务书 PDF 路径，仅写入 config_args_raw.json 便于实验记录（程序不读取 PDF）。',
    )
    parser.add_argument(
        '--no_thesis_metric_log', action='store_true',
        help='关闭验证结束时打印任务书对齐指标（mAP@0.5、APs 等）。',
    )
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument(
        '--output_dir', '--output', '--out', '--save_dir', '--checkpoint_dir', '--results_dir',
        type=str,
        default='',
        dest='output_dir',
        metavar='PATH',
        help='模型与日志输出目录（checkpoint、config、info.txt 等）。未写时可用 DINO_OUTPUT_DIR 或 DINO_TRAIN_OUTPUT_DIR。',
    )
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument(
        '--resume',
        default='',
        help='指定 checkpoint 续训。留空且设置 --output_dir 时，自动使用该目录下 checkpoint.pth；'
             '若仅有 checkpointNNNN.pth（如死机未写完主文件），则自动选最大编号。',
    )
    parser.add_argument(
        '--no_auto_resume',
        action='store_true',
        help='禁止根据 --output_dir 自动续训（同目录有 checkpoint 也从头训）。',
    )
    parser.add_argument(
        '--resume_allow_partial',
        action='store_true',
        help='以 strict=False 续训：只加载名称与形状一致的参数；若有缺失键则不复原 optimizer/lr_scheduler，'
             '并将 start_epoch 置 0。用于 checkpoint 与当前 deformable DINO 不完全一致时（仅部分层可对上）。',
    )
    parser.add_argument(
        '--emergency_checkpoint_iters', type=int, default=None, metavar='N',
        help='覆盖配置：每 N 个训练 iteration 写 checkpoint_emergency.pth（易死机时改小，如 30）；0=关闭。',
    )
    parser.add_argument(
        '--pretrain_model_path', type=str, default='',
        help='COCO（或同架构）预训练 DINO 权重 .pth；仅在不使用 --resume 时加载，strict=False 部分加载。'
             '亦可用环境变量 DINO_PRETRAIN_MODEL_PATH 或 DINO_PRETRAIN；'
             '或在 util/thesis_spec.py 中设置 DEFAULT_DINO_COCO_PRETRAIN_PATH。',
    )
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--grad_accum_steps', default=1, type=int,
                        help='梯度累积步数（通常由 GPU 自适应自动覆盖）。')
    parser.add_argument('--distill_teacher_ckpt', default='', type=str,
                        help='Teacher checkpoint 路径；设置后启用知识蒸馏。')
    parser.add_argument('--distill_weight', default=0.0, type=float,
                        help='蒸馏损失权重。')
    parser.add_argument('--distill_temperature', default=4.0, type=float,
                        help='蒸馏 KL 温度。')
    parser.add_argument('--gpu_adaptive_safety_reserve_ratio', default=0.10, type=float,
                        help='GPU自适应探测时的显存预留比例（默认0.10，即预留10%）。')
    parser.add_argument('--gpu_adaptive_distill_overhead_gb', default=4.0, type=float,
                        help='开启蒸馏时在探测峰值上额外加的显存缓冲（GB，默认4.0）。')
    parser.add_argument('--no_h100_speedup', action='store_true',
                        help='关闭 H100 自动吞吐优化（默认开启，仅影响运行速度参数）。')
    parser.add_argument(
        '--laptop_stable', action='store_true',
        help='降低 DataLoader 线程与 cudnn 激进优化，减轻笔记本满载死机概率；亦可设环境变量 DINO_LAPTOP_STABLE=1。',
    )
    parser.add_argument(
        '--interactive', '-i', action='store_true',
        help='进入旧版完整交互（多步问答）：路径、运行模式、VisDrone 训练预设菜单；'
             '默认 TTY 下改为三步 Quick Launch（选 profile→路径→确认），不使用 -i 时优先 Quick Launch。',
    )
    parser.add_argument(
        '--no_auto_prompt', action='store_true',
        help='关闭「直接运行 main 时自动询问路径」；缺路径时将报错退出（适合无 TTY 的脚本）。',
    )
    parser.add_argument(
        '--no_training_menu', action='store_true',
        help='关闭路径交互后的「训练预设」菜单（显存档位、epoch、batch 等）。',
    )
    parser.add_argument(
        '--no_run_mode_menu', action='store_true',
        help='关闭路径交互后的「运行模式」菜单（训练 / 仅验证）；仍可用命令行 --eval。',
    )
    parser.add_argument(
        '--no_auto_num_classes', action='store_true',
        help='关闭根据 COCO 标注 categories 自动设置 num_classes/dn_labelbook_size（VisDrone 10 类需与 checkpoint 一致）。',
    )
    parser.add_argument(
        '--no_auto_ckpt_arch', action='store_true',
        help='关闭根据 --resume/输出目录 checkpoint 自动对齐 add_channel_attention 等结构（与训练时 thesis 配置一致）。',
    )

    return parser


def dataset_paths_ready(args):
    """是否已配置训练数据（命令行或环境变量展开后）。"""
    if (args.coco_path or '').strip():
        return True
    if (getattr(args, 'dataset_json_root', None) or '').strip():
        return True
    if args.train_ann_file and args.train_img_folder:
        return True
    return False


def needs_path_prompt(args):
    return (not (args.output_dir or '').strip()) or (not dataset_paths_ready(args))


def tty_primary_process():
    """可在终端交互的进程（单卡 / distributed rank 0）。"""
    if not sys.stdin.isatty():
        return False
    if os.environ.get('RANK', '0') != '0':
        return False
    return True


def _prompt_line(msg):
    try:
        return input(msg).strip().strip('"').strip("'")
    except EOFError:
        return ''


def prompt_paths_interactive(args, *, ask_visdrone_choice=True):
    """在 TTY 中询问输出目录与数据集路径。"""
    if not sys.stdin.isatty():
        raise SystemExit(
            '当前不是交互式终端（无 TTY）。请：在 cmd/PowerShell 里运行 python main.py；'
            '或使用 --output / --dataset_json_root 等参数、环境变量 DINO_OUTPUT_DIR 等。'
        )
    print('\n======== DINO 交互式路径 ========')
    if not (args.output_dir or '').strip():
        o = _prompt_line('① 输出目录（保存 checkpoint / 日志，必填）: ')
        if not o:
            raise SystemExit('[interactive] 未设置输出目录')
        args.output_dir = str(Path(o).expanduser())

    if not dataset_paths_ready(args):
        print('\n② 训练数据（二选一）')
        print('  1 = 标准 COCO：根目录含 train2017、val2017、annotations')
        print('  2 = COCO JSON + 图片文件夹（如 VisDrone 转 COCO 后的 out_root + images）')
        ch = (_prompt_line('请选择 [1/2]，默认 2: ') or '2').strip()
        if ch == '1':
            p = _prompt_line('   COCO 根目录: ')
            if not p:
                raise SystemExit('[interactive] 未输入 COCO 根目录')
            args.coco_path = str(Path(p).expanduser().resolve())
            args.dataset_json_root = ''
            args.dataset_images = ''
            args.train_ann_file = None
            args.val_ann_file = None
            args.train_img_folder = None
            args.val_img_folder = None
        else:
            jr = _prompt_line('   JSON 根目录（内含 annotations/instances_*.json）: ')
            im = _prompt_line('   图片文件夹（与标注中 file_name 一致）: ')
            if not jr or not im:
                raise SystemExit('[interactive] JSON 根目录与图片文件夹均需填写')
            args.dataset_json_root = str(Path(jr).expanduser().resolve())
            args.dataset_images = str(Path(im).expanduser().resolve())
            args.coco_path = ''

    if ask_visdrone_choice:
        vd = (_prompt_line('\n③ 使用 VisDrone 10 类配置 (DINO_4scale_visdrone_thesis.py)? [y/N]: ') or 'n').lower()
        if vd == 'y':
            cfg = Path(__file__).resolve().parent / 'config' / 'DINO' / 'DINO_4scale_visdrone_thesis.py'
            args.config_file = str(cfg)
    if not (args.pretrain_model_path or '').strip():
        pr = _prompt_line(
            '\n④ COCO 预训练 DINO 权重 .pth（可选；回车跳过。已设 DINO_PRETRAIN_MODEL_PATH 则通常不必填）\n   路径: '
        ).strip()
        if pr:
            args.pretrain_model_path = str(Path(pr).expanduser())
    print('================================\n')


def prompt_run_mode_interactive(args):
    """
    在已完成路径交互后询问：训练 或 仅验证（等价于命令行 --eval）。
    若已在命令行传入 --eval / --test，则不重复询问。
    """
    if args.eval or args.test:
        return
    if not tty_primary_process():
        return
    print('\n======== 运行模式 ========')
    print('  1 = 训练（默认）')
    print('  2 = 仅验证模型（在验证集上计算 COCO mAP；权重来自 --resume 或输出目录下 checkpoint）')
    print('  （非交互时可用: python main.py --eval --resume 权重.pth ...）')
    ch = (_prompt_line('请选择 [1/2]，默认 1: ') or '1').strip()
    if ch != '2':
        print('================================\n')
        return
    args.eval = True
    ck = _prompt_line(
        'checkpoint 路径（回车则自动从「输出目录」查找 checkpoint.pth / checkpointNNNN.pth）: '
    ).strip()
    if ck:
        args.resume = str(Path(ck).expanduser())
    print('已选择：仅验证（--eval）')
    print('================================\n')


def apply_output_dir_from_env(args):
    if (getattr(args, 'output_dir', None) or '').strip():
        return
    for key in ('DINO_OUTPUT_DIR', 'DINO_TRAIN_OUTPUT_DIR'):
        env = (os.environ.get(key) or '').strip()
        if env:
            args.output_dir = env
            return


def apply_pretrain_path_from_env(args):
    """命令行未指定时，从环境变量读取预训练权重路径。"""
    if (getattr(args, 'pretrain_model_path', None) or '').strip():
        return
    for key in ('DINO_PRETRAIN_MODEL_PATH', 'DINO_PRETRAIN'):
        v = (os.environ.get(key) or '').strip()
        if v:
            args.pretrain_model_path = v
            return


def apply_pretrain_path_from_thesis_default(args):
    """若 thesis_spec 中配置了 DEFAULT_DINO_COCO_PRETRAIN_PATH 且文件存在，则采用。"""
    if (getattr(args, 'pretrain_model_path', None) or '').strip():
        return
    try:
        from util.thesis_spec import DEFAULT_DINO_COCO_PRETRAIN_PATH
    except ImportError:
        return
    p = (DEFAULT_DINO_COCO_PRETRAIN_PATH or '').strip()
    if not p:
        return
    p = str(Path(p).expanduser())
    if Path(p).is_file():
        args.pretrain_model_path = p


def apply_user_dataset_paths(args):
    """将 --dataset_json_root / --dataset_images 或环境变量解析为 train/val 标注与图像路径。"""
    from util.visdrone_dataset_pack import resolve_dataset_json_root_and_images

    resolve_dataset_json_root_and_images(args)


def folder_has_visdrone_cleaned_txt(root):
    root = Path(root)
    if not root.is_dir():
        return False
    return any(root.rglob('*_cleaned.txt'))


def normalize_visdrone_data_folder(args):
    """
    若 --dataset/--coco_path 指向仅有 *_cleaned.txt、尚无标准 COCO 布局的目录，
    自动改为 dataset_json_root（随后会生成 annotations/*.json）。
    """
    if getattr(args, 'no_auto_visdrone_coco', False):
        return
    if (getattr(args, 'dataset_json_root', None) or '').strip():
        return
    coco = (getattr(args, 'coco_path', None) or '').strip()
    if not coco:
        return
    root = Path(coco).expanduser().resolve()
    if not root.is_dir():
        return
    if (root / 'train2017').is_dir():
        return
    if (root / 'annotations' / 'instances_train2017.json').is_file():
        from util.visdrone_dataset_pack import is_visdrone_self_contained_pack

        if is_visdrone_self_contained_pack(root):
            args.dataset_json_root = str(root)
            args.coco_path = ''
            print(
                '[visdrone] 检测到自包含数据包（annotations/ + images/，无 train2017/）。'
                '已切换为 --dataset_json_root 模式；图像默认使用包内 images/。'
            )
        return
    if not folder_has_visdrone_cleaned_txt(root):
        return
    args.dataset_json_root = str(root)
    args.coco_path = ''
    print(
        f'[visdrone] 数据目录识别为 VisDrone 处理文件夹（将自动生成 COCO）: {root}'
    )


def ensure_visdrone_coco_json_files(args):
    """若 train 标注 JSON 不存在但目录内有 *_cleaned.txt，则自动生成。"""
    if getattr(args, 'no_auto_visdrone_coco', False):
        return
    train_ann = getattr(args, 'train_ann_file', None)
    if not train_ann:
        return
    p = Path(train_ann)
    if p.is_file():
        return
    if p.parent.name != 'annotations':
        return
    root = p.parent.parent
    if not folder_has_visdrone_cleaned_txt(root):
        return
    from util.thesis_spec import default_thesis_image_folder
    from tools.visdrone_cleaned_txt_to_coco import write_visdrone_coco_to_out_root

    tif = getattr(args, 'train_img_folder', None)
    img = (str(tif).strip() if tif else '') or (os.environ.get('THESIS_IMAGES_DIR') or '').strip()
    if not img:
        img = default_thesis_image_folder()
    img = str(Path(img).expanduser().resolve())
    args.train_img_folder = img
    if getattr(args, 'val_img_folder', None) is None:
        args.val_img_folder = img
    vrat = float(os.environ.get('VISDRONE_AUTO_VAL_RATIO', '0.2'))
    print(
        f'[visdrone] 未找到 {p.name}，正从 txt 与图片目录生成 COCO：\n'
        f'         txt={root}\n         images={img}'
    )
    try:
        write_visdrone_coco_to_out_root(
            img, root, root, val_ratio=vrat, seed=getattr(args, 'seed', 42),
        )
    except ValueError as err:
        raise SystemExit(
            f'{err}\n'
            '请确认原始图片目录正确，并用 --dataset_images 或环境变量 '
            'DINO_DATASET_IMAGES / THESIS_IMAGES_DIR 指定（需与 txt 中图片主文件名一致）。'
        ) from err
    if not p.is_file():
        raise RuntimeError(f'自动生成 COCO 失败: {p}')


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def maybe_apply_h100_speed_profile(args, device, logger=None):
    """H100 运行时加速，仅改吞吐相关参数，不改精度/结构相关超参。"""
    if getattr(args, 'no_h100_speedup', False):
        return
    if device.type != 'cuda':
        return
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(idx)
    except Exception:
        return
    if 'H100' not in str(gpu_name).upper():
        return

    cpu_count = int(os.cpu_count() or 8)
    args.num_workers = max(int(getattr(args, 'num_workers', 8) or 8), min(16, cpu_count))
    if int(getattr(args, 'cuda_empty_cache_every_iters', 0) or 0) > 0:
        args.cuda_empty_cache_every_iters = 0
    args.dataloader_pin_memory = True
    args.dataloader_persistent_workers = args.num_workers > 0
    args.dataloader_prefetch_factor = 4 if args.num_workers > 0 else None

    if logger:
        logger.info(
            '[H100加速] GPU=%s; num_workers=%s; pin_memory=%s; persistent_workers=%s; '
            'prefetch_factor=%s; cuda_empty_cache_every_iters=%s',
            gpu_name,
            args.num_workers,
            args.dataloader_pin_memory,
            args.dataloader_persistent_workers,
            args.dataloader_prefetch_factor,
            args.cuda_empty_cache_every_iters,
        )

def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0 and args.output_dir:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    if getattr(args, 'emergency_checkpoint_iters', None) is not None:
        args.checkpoint_every_iters = int(args.emergency_checkpoint_iters)

    if getattr(args, 'thesis_taskbook_pdf', None):
        setattr(args, 'thesis_taskbook_pdf_resolved', str(Path(args.thesis_taskbook_pdf).expanduser()))
    args.log_thesis_metrics = (getattr(args, 'num_classes', 91) == 10) and (not getattr(args, 'no_thesis_metric_log', False))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False
    if not hasattr(args, 'gradient_accumulation_steps'):
        args.gradient_accumulation_steps = int(getattr(args, 'grad_accum_steps', 1) or 1)

    # setup logger (no log files if --output_dir omitted)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logger = setup_logger(
            output=os.path.join(args.output_dir, 'info.txt'),
            distributed_rank=args.rank, color=False, name="detr")
    else:
        logger = setup_logger(output=None, distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0 and args.output_dir:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    if getattr(args, 'laptop_stable', False):
        args.laptop_stable_training = True
    apply_laptop_stable_training(args, logger)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # 与分布式 local_rank 对齐：避免 torch.device('cuda') 与 torch.cuda.set_device 不一致
    device = torch.device(args.device)
    if device.type == 'cuda':
        if getattr(args, 'distributed', False):
            device = torch.device(f'cuda:{int(args.gpu)}')
        elif device.index is None:
            device = torch.device(f'cuda:{torch.cuda.current_device()}')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 与交互菜单 / epoch / batch 无关：仅检测 MSDA CUDA 是否与当前 GPU 匹配
    from util.msda_preflight import run_msda_cuda_preflight
    run_msda_cuda_preflight(device, logger)

    # ---- GPU 自适应（最坏情况前向+反向探测，自动配 batch + 梯度累积）----
    if (not getattr(args, "eval", False)) and (not getattr(args, "test", False)):
        try:
            from util.gpu_adaptive import auto_adapt_batch
            auto_adapt_batch(
                args=args,
                build_model_fn=build_model_main,
                device=device,
                logger=logger,
            )
        except Exception as _e:
            if logger and utils.is_main_process():
                logger.warning("GPU 自适应探测失败，继续按原 batch_size 训练。错误: %s", _e)
    maybe_apply_h100_speed_profile(args, device, logger)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    log_gpu("Student构建完成")

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    teacher_model = None
    if (not args.eval) and (not args.test):
        teacher_model = setup_distillation(args, device, build_model_main)
        if teacher_model is not None:
            log_gpu("Teacher加载完成")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    use_class_balance = bool(getattr(args, 'train_class_balance_sampler', False))
    if use_class_balance and args.distributed:
        if args.rank == 0:
            logger.warning(
                'train_class_balance_sampler 与 DistributedSampler 不兼容，已忽略（请单卡训练时启用）。'
            )
        use_class_balance = False

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    elif use_class_balance:
        from datasets.coco_balance import compute_train_image_weights_coco
        from torch.utils.data import WeightedRandomSampler

        _w = compute_train_image_weights_coco(
            dataset_train,
            small_area_max=getattr(args, 'train_balance_small_area_max', None),
            small_boost_scale=float(getattr(args, 'train_balance_small_boost_scale', 0) or 0),
        )
        sampler_train = WeightedRandomSampler(
            weights=_w, num_samples=len(_w), replacement=True
        )
        if args.rank == 0:
            logger.info(
                'train_class_balance_sampler: WeightedRandomSampler，len=%s；'
                'small_boost(area_max=%s, scale=%s)。',
                len(_w),
                getattr(args, 'train_balance_small_area_max', None),
                getattr(args, 'train_balance_small_boost_scale', 0),
            )
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    _loader_common = {}
    if getattr(args, 'dataloader_pin_memory', False):
        _loader_common['pin_memory'] = True
    if getattr(args, 'dataloader_persistent_workers', False):
        _loader_common['persistent_workers'] = True
    _pf = getattr(args, 'dataloader_prefetch_factor', None)
    if _pf is not None and int(getattr(args, 'num_workers', 0) or 0) > 0:
        _loader_common['prefetch_factor'] = int(_pf)

    if args.distributed or not use_class_balance:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(
            dataset_train, batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn, num_workers=args.num_workers, **_loader_common)
    else:
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler=sampler_train,
            drop_last=True,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
            **_loader_common,
        )
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, **_loader_common)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = utils.torch_load_trusted(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir) if args.output_dir else Path('.')
    if (
        args.output_dir
        and not (args.resume or '').strip()
        and not getattr(args, 'no_auto_resume', False)
    ):
        ck_auto = utils.find_checkpoint_for_resume(args.output_dir)
        if ck_auto:
            args.resume = ck_auto
            logger.info('Auto-resume from: %s', ck_auto)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = utils.torch_load_trusted(args.resume, map_location='cpu')
        _ckpt_sd = utils.adapt_dino_checkpoint_state_dict(model_without_ddp, checkpoint['model'])
        _has_segm = utils.dino_ckpt_has_segm_heads(_ckpt_sd.keys())
        _is_segm_model = getattr(model_without_ddp, 'mask_head', None) is not None
        _resume_strict = not (_is_segm_model and not _has_segm)
        _mismatch_hint = utils.resume_checkpoint_vs_model_mismatch_hint(_ckpt_sd.keys(), model_without_ddp)
        _skip_optimizer_after_resume = False

        if getattr(args, 'resume_allow_partial', False):
            _resume_strict = False
            if args.rank == 0:
                logger.warning('--resume_allow_partial: loading with strict=False')

        if not _resume_strict and args.rank == 0 and (_is_segm_model and not _has_segm):
            logger.info(
                'Resume: checkpoint is plain detection weights (no mask heads); '
                'loading into segmentation wrapper with strict=False — mask branch stays randomly initialized.'
            )

        if _resume_strict:
            try:
                _incomp = model_without_ddp.load_state_dict(_ckpt_sd, strict=True)
            except RuntimeError as _e:
                if args.rank == 0:
                    if _mismatch_hint:
                        logger.error(_mismatch_hint)
                    logger.error(
                        '续训 strict 加载失败。若确认只需加载能对上的层（例如仅 backbone），可加上 '
                        '--resume_allow_partial（将放弃还原 optimizer/LR，start_epoch 从 0 开始）。'
                    )
                raise _e
            if args.rank == 0 and (len(_incomp.missing_keys) or len(_incomp.unexpected_keys)):
                logger.info(
                    'load_state_dict: missing %d, unexpected %d',
                    len(_incomp.missing_keys), len(_incomp.unexpected_keys),
                )
        else:
            _ckpt_sd_filt, _resume_shape_skip = utils.filter_state_dict_matching_shapes(
                model_without_ddp, _ckpt_sd
            )
            if args.rank == 0 and _resume_shape_skip:
                logger.warning(
                    'resume strict=False: skipped %d shape-mismatch keys (e.g. num_classes): %s',
                    len(_resume_shape_skip),
                    _resume_shape_skip[:16],
                )
            _incomp = model_without_ddp.load_state_dict(_ckpt_sd_filt, strict=False)
            if args.rank == 0 and (len(_incomp.missing_keys) or len(_incomp.unexpected_keys)):
                logger.warning(
                    'load_state_dict strict=False: missing %d, unexpected %d',
                    len(_incomp.missing_keys), len(_incomp.unexpected_keys),
                )
            _missing = _incomp.missing_keys
            _segm_only_missing = _is_segm_model and not _has_segm and _missing and all(
                m.startswith('bbox_attention.') or m.startswith('mask_head.') for m in _missing
            )
            if getattr(args, 'resume_allow_partial', False) and len(_missing) > 0 and not _segm_only_missing:
                _skip_optimizer_after_resume = True
                if args.rank == 0 and _mismatch_hint:
                    logger.warning(_mismatch_hint)

        if args.use_ema:
            if 'ema_model' in checkpoint:
                _ema_sd = utils.adapt_dino_checkpoint_state_dict(ema_m.module, checkpoint['ema_model'])
                _has_segm_e = utils.dino_ckpt_has_segm_heads(_ema_sd.keys())
                _ema_strict = not (_is_segm_model and not _has_segm_e)
                if getattr(args, 'resume_allow_partial', False):
                    _ema_strict = False
                if not _ema_strict:
                    _ema_sd, _ = utils.filter_state_dict_matching_shapes(ema_m.module, _ema_sd)
                ema_m.module.load_state_dict(_ema_sd, strict=_ema_strict)
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)

        if (
            not args.eval
            and not _skip_optimizer_after_resume
            and 'optimizer' in checkpoint
            and 'lr_scheduler' in checkpoint
            and 'epoch' in checkpoint
        ):
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if checkpoint.get('intra_epoch_checkpoint'):
                args.start_epoch = int(checkpoint['epoch'])
                if args.rank == 0:
                    logger.warning(
                        '从 checkpoint_emergency 续训：将重新完整跑 epoch %s（该 epoch 内已训过的 batch 会再过一遍）。',
                        args.start_epoch,
                    )
            else:
                args.start_epoch = checkpoint['epoch'] + 1
        elif _skip_optimizer_after_resume and args.rank == 0:
            logger.warning(
                'Partial resume: 未加载 optimizer / lr_scheduler，start_epoch 已置为 0（与随机初始化的未匹配层一致）。'
            )
            args.start_epoch = 0

    if (not args.resume) and (args.pretrain_model_path or '').strip():
        from collections import OrderedDict

        pre_path = str(Path(args.pretrain_model_path).expanduser().resolve())
        if not Path(pre_path).is_file():
            if args.rank == 0:
                logger.warning('pretrain_model_path 不是有效文件，跳过加载: %s', pre_path)
        else:
            raw = utils.torch_load_trusted(pre_path, map_location='cpu')
            try:
                ckpt_full, state_dict = utils.split_checkpoint_and_model_state(raw)
            except TypeError as err:
                if args.rank == 0:
                    logger.error('预训练文件格式错误（需 dict，且含 model / state_dict 或为纯 state_dict）: %s', err)
                raise
            _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
            ignorelist = []

            def check_keep(keyname, ignorekeywordlist):
                for keyword in ignorekeywordlist:
                    if keyword in keyname:
                        ignorelist.append(keyname)
                        return False
                return True

            _ckpt_sd = utils.adapt_dino_checkpoint_state_dict(model_without_ddp, state_dict)
            if args.rank == 0:
                logger.info('Pretrain weights: %s', pre_path)
                logger.info('Ignore keys: %s', json.dumps(ignorelist, indent=2))
            _tmp_st = OrderedDict(
                {k: v for k, v in _ckpt_sd.items() if check_keep(k, _ignorekeywordlist)}
            )
            _tmp_st, _shape_skip = utils.filter_state_dict_matching_shapes(
                model_without_ddp, _tmp_st
            )
            if args.rank == 0 and _shape_skip:
                _show = _shape_skip[:24]
                _extra = '' if len(_shape_skip) <= 24 else f' …(+{len(_shape_skip) - 24} more)'
                logger.warning(
                    'pretrain: %d keys skipped (shape mismatch vs current model, e.g. 91→10 classes): %s%s',
                    len(_shape_skip),
                    _show,
                    _extra,
                )
            _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
            if args.rank == 0:
                logger.warning(
                    'pretrain load strict=False: missing %d, unexpected %d',
                    len(_load_output.missing_keys),
                    len(_load_output.unexpected_keys),
                )

            if args.use_ema:
                if isinstance(ckpt_full, dict) and 'ema_model' in ckpt_full:
                    _ema_sd = utils.adapt_dino_checkpoint_state_dict(ema_m.module, ckpt_full['ema_model'])
                    _ema_sd, _ema_shape_skip = utils.filter_state_dict_matching_shapes(
                        ema_m.module, _ema_sd
                    )
                    if args.rank == 0 and _ema_shape_skip:
                        logger.warning(
                            'pretrain EMA: skipped %d shape-mismatch keys',
                            len(_ema_shape_skip),
                        )
                    _ema_out = ema_m.module.load_state_dict(_ema_sd, strict=False)
                    if args.rank == 0:
                        logger.info(
                            'pretrain EMA: missing %d, unexpected %d',
                            len(_ema_out.missing_keys),
                            len(_ema_out.unexpected_keys),
                        )
                else:
                    ema_m.set(model_without_ddp)
                    if args.rank == 0:
                        logger.info('pretrain checkpoint 无 ema_model，已将 EMA 同步为当前加载后的模型权重')

    _wu_ep = int(getattr(args, 'lr_warmup_epochs', 0) or 0)
    if _wu_ep > 0 and not args.onecyclelr and not args.eval:
        raw_lrs = [float(g['lr']) for g in optimizer.param_groups]
        if args.start_epoch >= _wu_ep:
            args._warmup_target_lrs = None
        elif args.resume and args.start_epoch > 0:
            inv = float(args.start_epoch) / float(_wu_ep)
            args._warmup_target_lrs = [u / inv for u in raw_lrs]
        else:
            args._warmup_target_lrs = raw_lrs
        if args.rank == 0 and args._warmup_target_lrs is not None:
            logger.info('LR warmup: %s epochs (linear ramp to base lrs)', _wu_ep)
    else:
        args._warmup_target_lrs = None

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    print("Start training")
    if not args.output_dir:
        logger.warning(
            "No --output_dir given: training runs but checkpoints/logs are not saved. "
            "Use e.g. --output_dir F:/runs/my_exp")
    _cei = int(getattr(args, 'checkpoint_every_iters', 0) or 0)
    if _cei > 0 and args.output_dir and args.rank == 0:
        logger.info(
            'checkpoint_every_iters=%s：训练中将周期性写入 %s（适合整 epoch 跑不完、易死机场景）。',
            _cei,
            os.path.join(args.output_dir, 'checkpoint_emergency.pth'),
        )
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        _wu_ep = int(getattr(args, 'lr_warmup_epochs', 0) or 0)
        wlist = getattr(args, '_warmup_target_lrs', None)
        if wlist is not None and not args.onecyclelr and epoch < _wu_ep:
            wf = float(epoch + 1) / float(_wu_ep)
            for pg, tlr in zip(optimizer.param_groups, wlist):
                pg['lr'] = tlr * wf
        try:
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args,
                logger=logger, ema_m=ema_m, model_without_ddp=model_without_ddp, teacher_model=teacher_model)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise e
            old_bs = int(getattr(args, 'batch_size', 1) or 1)
            new_bs = max(1, old_bs // 2)
            if new_bs == old_bs:
                raise e
            logger.warning('[OOM恢复] epoch %s OOM，batch_size %s -> %s', epoch, old_bs, new_bs)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            args.batch_size = new_bs
            args.gradient_accumulation_steps = int(getattr(args, 'gradient_accumulation_steps', 1) or 1) * 2
            args.grad_accum_steps = args.gradient_accumulation_steps
            if args.distributed or not use_class_balance:
                batch_sampler_train = torch.utils.data.BatchSampler(
                    sampler_train, args.batch_size, drop_last=True)
                data_loader_train = DataLoader(
                    dataset_train, batch_sampler=batch_sampler_train,
                    collate_fn=utils.collate_fn, num_workers=args.num_workers, **_loader_common)
            else:
                data_loader_train = DataLoader(
                    dataset_train,
                    batch_size=args.batch_size,
                    sampler=sampler_train,
                    drop_last=True,
                    collate_fn=utils.collate_fn,
                    num_workers=args.num_workers,
                    **_loader_common,
                )
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args,
                logger=logger, ema_m=ema_m, model_without_ddp=model_without_ddp, teacher_model=teacher_model)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            if _wu_ep <= 0 or epoch >= _wu_ep:
                lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)
            utils.save_on_master(weights, output_dir / 'checkpoint_latest.pth')
                
        # eval
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # eval ema
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())

        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    try:
        efficiency_report(ema_m.module if args.use_ema else model_without_ddp, device)
    except Exception as _e:
        logger.warning('efficiency_report skipped: %s', _e)

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script',
        parents=[get_args_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Dataset / output examples:\n'
            '  标准 COCO 目录:  python main.py --dataset YOUR_COCO_ROOT --output YOUR_OUT_DIR\n'
            '  仅 VisDrone 处理文件夹（含 *_cleaned.txt）:  python main.py --dataset THAT_FOLDER --output YOUR_OUT_DIR\n'
            '    （自动生成 annotations/*.json；图片默认用 thesis_spec 中的 origin，否则加 --dataset_images）\n'
            '  JSON 已存在时:  python main.py --dataset_json_root YOUR_OUT_ROOT --dataset_images YOUR_IMG_DIR --output YOUR_OUT_DIR\n'
            '  或逐项指定:  python main.py --train_img_folder YOUR_IMG_DIR --train_ann_file YOUR_TRAIN.json '
            '--val_ann_file YOUR_VAL.json --output_dir YOUR_OUT_DIR\n'
            '  环境变量: DINO_DATASET_JSON_ROOT, DINO_DATASET_IMAGES, DINO_OUTPUT_DIR\n'
            '  输出目录:  --output F:/runs/exp1  （与 --output_dir 相同）\n'
            '  VisDrone 论文预设:  python main.py --thesis_visdrone_50 --amp --output YOUR_OUT_DIR\n'
            '  VisDrone 10 类权重/数据: 使用 config/DINO/DINO_4scale_visdrone_thesis.py，或让程序从标注 JSON 自动 num_classes=10（见 --no_auto_num_classes）。\n'
            '  毕业论文 PDF 提 AP（EMA+论文栈）:  config/DINO/DINO_4scale_visdrone_graduation_max_ap.py（交互菜单「长训+EMA」）。\n'
            '    （40GB A100/Colab 建议始终加 --amp；仍显存不足时 --options batch_size=2）\n'
            '    （MSDA：菜单③ 分 A100/G4/H100；无菜单时 DINO_MSDA_INSTALL_PROFILE=a100|g4|h100）\n'
            '  终端交互: 直接 python main.py → 默认「三步 Quick Launch」（选 profile→路径→确认）；'
            '旧版多步菜单: python main.py -i 。跳过 Quick Launch: DINO_NO_QUICK_LAUNCH=1 或非默认 -c 。\n'
            '  --no_run_mode_menu / --no_training_menu 仅作用于旧版 -i 流程。\n'
        ),
    )
    args = parser.parse_args()
    normalize_visdrone_data_folder(args)
    if args.thesis_visdrone_50:
        from util.thesis_spec import (
            DEFAULT_THESIS_COCO_ROOT,
            DEFAULT_THESIS_TXT_DIR,
            default_thesis_image_folder,
        )
        thesis_cfg = Path(__file__).resolve().parent / 'config' / 'DINO' / 'DINO_4scale_visdrone_thesis.py'
        args.config_file = str(thesis_cfg)
        coco_root = os.environ.get('THESIS_COCO_ROOT', DEFAULT_THESIS_COCO_ROOT).strip() or DEFAULT_THESIS_COCO_ROOT
        img_dir = os.environ.get('THESIS_IMAGES_DIR', '').strip() or default_thesis_image_folder()
        ann_train = Path(coco_root) / 'annotations' / 'instances_train2017.json'
        ann_val = Path(coco_root) / 'annotations' / 'instances_val2017.json'
        if args.train_ann_file is None:
            args.train_ann_file = str(ann_train)
        if args.val_ann_file is None:
            args.val_ann_file = str(ann_val)
        if args.train_img_folder is None:
            args.train_img_folder = img_dir
        if args.val_img_folder is None:
            args.val_img_folder = img_dir
        args.coco_path = ''
        if not Path(args.train_ann_file).is_file():
            raise SystemExit(
                f'[thesis_visdrone_50] 未找到 COCO 标注: {args.train_ann_file}\n'
                '请先运行（按你的图像目录改 --images_dir）：\n'
                '  python tools/visdrone_cleaned_txt_to_coco.py '
                f'--images_dir "{img_dir}" '
                f'--txt_dir "{DEFAULT_THESIS_TXT_DIR}" '
                f'--out_root "{coco_root}" --val_ratio 0.2\n'
                '若图像不在上述路径，请设置环境变量 THESIS_IMAGES_DIR 或命令行指定 --train_img_folder。'
            )
    # 先应用环境变量，再交互询问，避免重复索要已用环境变量提供的路径
    apply_user_dataset_paths(args)
    apply_output_dir_from_env(args)
    apply_pretrain_path_from_env(args)

    quick_launch_done = False
    if (
        (not args.no_auto_prompt)
        and tty_primary_process()
        and (not args.interactive)
        and (not args.eval)
        and (not args.test)
    ):
        from util.quick_launch import quick_launch

        quick_launch_done = quick_launch(args)

    if not quick_launch_done:
        auto_prompt = (
            (not args.no_auto_prompt)
            and needs_path_prompt(args)
            and tty_primary_process()
        )
        did_path_prompt = False
        if args.interactive:
            prompt_paths_interactive(args, ask_visdrone_choice=True)
            apply_user_dataset_paths(args)
            did_path_prompt = True
        elif auto_prompt:
            prompt_paths_interactive(args, ask_visdrone_choice=True)
            apply_user_dataset_paths(args)
            did_path_prompt = True
        elif needs_path_prompt(args):
            raise SystemExit(
                '未设置输出目录或数据集路径。\n'
                '  • 在终端运行: python main.py  （默认三步 Quick Launch；旧版菜单请加 -i）\n'
                '  • 或命令行: --output ... 与 --dataset / --dataset_json_root + --dataset_images 等\n'
                '  • 或环境变量: DINO_OUTPUT_DIR, DINO_DATASET_JSON_ROOT, DINO_DATASET_IMAGES\n'
                '无 TTY 的自动化场景请加: --no_auto_prompt 并自行传齐路径。\n'
                '跳过 Quick Launch: 环境变量 DINO_NO_QUICK_LAUNCH=1 或使用非默认 -c 配置文件。'
            )

        ensure_visdrone_coco_json_files(args)

        if (
            (did_path_prompt or args.interactive)
            and tty_primary_process()
            and not getattr(args, 'no_run_mode_menu', False)
            and not args.test
        ):
            prompt_run_mode_interactive(args)

        _force_menu = (os.environ.get('DINO_TRAINING_MENU', '').strip().lower() in ('1', 'true', 'yes', 'y'))
        if (
            (did_path_prompt or args.interactive or _force_menu)
            and tty_primary_process()
            and not getattr(args, 'no_training_menu', False)
            and not args.eval
            and not args.test
        ):
            from util.training_interactive_menu import prompt_visdrone_training_presets
            prompt_visdrone_training_presets(args)
    else:
        ensure_visdrone_coco_json_files(args)

    apply_pretrain_path_from_thesis_default(args)

    from util.coco_ann_checks import maybe_align_num_classes_from_dataset
    maybe_align_num_classes_from_dataset(args)
    from util.checkpoint_arch_alignment import maybe_align_options_from_checkpoint_arch
    maybe_align_options_from_checkpoint_arch(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
