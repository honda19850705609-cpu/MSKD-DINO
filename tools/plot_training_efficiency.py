#!/usr/bin/env python3
"""
从训练输出目录读取 log.txt 与 checkpoint 时间戳，绘制效率/收敛分析图（matplotlib）。

示例:
  python tools/plot_training_efficiency.py
  python tools/plot_training_efficiency.py --result_dir F:/paper/VisDrone_processing/...
  python tools/plot_training_efficiency.py --curves_only --result_dir <训练输出目录>

省略 --result_dir 时顺序尝试: 环境变量 DINO_TRAIN_OUTPUT_DIR、DINO_OUTPUT_DIR、
thesis_spec.DEFAULT_FULL_TRAIN_OUTPUT_DIR（若含 log.txt）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _setup_matplotlib_zh():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fam in ('Microsoft YaHei', 'SimHei', 'DejaVu Sans'):
        try:
            plt.rcParams['font.sans-serif'] = [fam]
            plt.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue
    return plt


def parse_log(log_path: str) -> tuple[list[dict], list[int]]:
    rows = []
    epochs = []
    with open(log_path, encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
                epochs.append(i)
            except json.JSONDecodeError:
                continue
    return rows, epochs


def checkpoint_epoch_seconds(result_dir: str) -> tuple[list[int], list[float]]:
    """按 checkpointNNNN.pth 修改时间估计每个 epoch 耗时；仅当 epoch 编号连续相邻时才取间隔（避免 0010→0099 误判）。"""
    pat = re.compile(r'checkpoint(\d{4})\.pth$')
    items = []
    for name in os.listdir(result_dir):
        m = pat.match(name)
        if not m:
            continue
        ep = int(m.group(1))
        p = os.path.join(result_dir, name)
        items.append((ep, os.path.getmtime(p)))
    items.sort(key=lambda x: x[0])
    if len(items) < 2:
        return [], []
    epoch_ids = []
    secs = []
    for j in range(1, len(items)):
        ep_prev, t_prev = items[j - 1]
        ep_cur, t_cur = items[j]
        if ep_cur != ep_prev + 1:
            continue
        dt = t_cur - t_prev
        if dt <= 0 or dt > 86400 * 2:
            continue
        epoch_ids.append(ep_cur)
        secs.append(dt)
    return epoch_ids, secs


def _epoch_x(rows: list[dict]) -> list[int]:
    """横轴：第几轮（1-based），优先用 log 里 epoch 字段。"""
    out = []
    for i, r in enumerate(rows):
        if 'epoch' in r:
            out.append(int(r['epoch']) + 1)
        else:
            out.append(i + 1)
    return out


def _write_text_report(
    path: str,
    result_dir: str,
    rows: list[dict],
    map50: list,
    map5095: list,
    aps_list: list,
    apm_list: list,
    apl_list: list,
    secs: list,
    nparam: str | None,
) -> None:
    lines = [
        '训练结果分析（自动生成）',
        f'结果目录: {result_dir}',
        f'记录轮数: {len(rows)}',
        '',
    ]
    valid50 = [(x, v) for x, v in zip(_epoch_x(rows), map50) if v is not None and v > 0]
    valid95 = [(x, v) for x, v in zip(_epoch_x(rows), map5095) if v is not None and v > 0]
    if valid50:
        best_x, best_v = max(valid50, key=lambda t: t[1])
        lines.append(f'最佳 mAP@0.5: {best_v:.4f}（第 {best_x} 轮记录）')
    if valid95:
        best_x, best_v = max(valid95, key=lambda t: t[1])
        lines.append(f'最佳 mAP@[.5:.95]: {best_v:.4f}（第 {best_x} 轮记录）')
    xs = _epoch_x(rows)
    for name, series in (
        ('AP small', aps_list),
        ('AP medium', apm_list),
        ('AP large', apl_list),
    ):
        valid_s = [(x, v) for x, v in zip(xs, series) if v is not None and v > 0]
        if valid_s:
            bx, bv = max(valid_s, key=lambda t: t[1])
            lines.append(f'最佳 {name}: {bv:.4f}（第 {bx} 轮记录）')
    last = rows[-1] if rows else {}
    if last:
        lines.append(f'最后一轮 train_loss: {last.get("train_loss", "N/A")}')
        lines.append(f'最后一轮 test_loss: {last.get("test_loss", "N/A")}')
    lines.append('')
    if secs:
        import numpy as np
        lines.append(f'相邻 checkpoint 估计单轮耗时: 平均 {float(np.mean(secs)):.1f} s，合计约 {float(np.sum(secs))/60:.1f} min（{len(secs)} 个间隔）')
    else:
        lines.append('未得到相邻 checkpoint 时间间隔（或编号不连续）。')
    if nparam:
        lines.append(f'参数量（info.txt）: {nparam}')
    nz50 = sum(1 for v in map50 if v is not None and float(v) > 1e-4)
    if len(rows) > 5 and nz50 <= max(2, len(rows) // 20):
        lines.extend([
            '',
            '【注意】log 中绝大部分 epoch 的 COCO bbox AP 为 0 或近 0，'
            '若你期望有正常 mAP，请检查：类别数与 JSON、评测 useCats、是否误用另一份 log、'
            '或仅用 --eval 时 checkpoint 是否匹配。',
        ])
    lines.extend([
        '',
        '说明: 若前期验证 mAP 全为 0，多为评测尚未产生有效检测；以中后期曲线为准。',
        '效率图: 单轮耗时 = 相邻 checkpointNNNN.pth 的修改时间差（含训练+验证+写盘）。',
    ])
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--result_dir',
        type=str,
        default='',
        help='含 log.txt 的训练输出目录；省略则用 DINO_TRAIN_OUTPUT_DIR / thesis 默认主实验目录',
    )
    ap.add_argument('--log_name', type=str, default='log.txt')
    ap.add_argument(
        '--plots_dir',
        type=str,
        default='',
        help='图表与报告输出目录；默认在 result_dir 下新建 training_analysis_report',
    )
    ap.add_argument(
        '--curves_only',
        action='store_true',
        help='仅保存曲线类 PNG（损失、mAP、尺度 AP、epoch 耗时），不写 analysis_report.txt，不生成文字摘要图与总览拼图。',
    )
    args = ap.parse_args()
    from util.thesis_spec import resolved_train_result_dir

    result_dir = os.path.abspath(str(resolved_train_result_dir(args.result_dir)))
    log_path = os.path.join(result_dir, args.log_name)
    if not os.path.isfile(log_path):
        raise SystemExit(f'未找到: {log_path}')

    plots_dir = (args.plots_dir or '').strip()
    if not plots_dir:
        plots_dir = os.path.join(
            result_dir, 'training_curves' if args.curves_only else 'training_analysis_report'
        )
    os.makedirs(plots_dir, exist_ok=True)
    plots_dir = os.path.abspath(plots_dir)

    rows, ep_idx = parse_log(log_path)
    if not rows:
        raise SystemExit('log.txt 无有效 JSON 行')

    train_loss = [r.get('train_loss') for r in rows]
    test_loss = [r.get('test_loss') for r in rows]
    lr = [r.get('train_lr') for r in rows]
    map5095 = []
    map50 = []
    aps_list = []
    apm_list = []
    apl_list = []
    for r in rows:
        c = r.get('test_coco_eval_bbox')
        if isinstance(c, list) and len(c) >= 2:
            map5095.append(c[0])
            map50.append(c[1])
            aps_list.append(c[3] if len(c) > 3 else None)
            apm_list.append(c[4] if len(c) > 4 else None)
            apl_list.append(c[5] if len(c) > 5 else None)
        else:
            map5095.append(None)
            map50.append(None)
            aps_list.append(None)
            apm_list.append(None)
            apl_list.append(None)

    x = _epoch_x(rows)
    plt = _setup_matplotlib_zh()
    import matplotlib.pyplot as plt_m
    import numpy as np

    # --- 图1: 损失 + 学习率 ---
    fig1, ax1 = plt_m.subplots(figsize=(9, 4.5))
    ax1.plot(x, train_loss, 'b-o', markersize=3, label='train_loss')
    if any(v is not None for v in test_loss):
        ax1.plot(x, test_loss, 'r-s', markersize=3, label='test_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练/验证损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, lr, 'g--', alpha=0.7, label='lr')
    ax2.set_ylabel('Learning rate')
    ax2.legend(loc='upper right')
    fig1.tight_layout()
    p1 = os.path.join(plots_dir, 'efficiency_loss_lr.png')
    fig1.savefig(p1, dpi=150)
    plt_m.close(fig1)

    # --- 图2: COCO mAP ---
    fig2, ax = plt_m.subplots(figsize=(8, 4.5))
    if any(v is not None for v in map50):
        ax.plot(x, map50, 'm-o', markersize=4, label='mAP@0.5')
    if any(v is not None for v in map5095):
        ax.plot(x, map5095, 'c-s', markersize=4, label='mAP@[.5:.95]')
    ax.set_xlabel('Epoch（log 记录序号）')
    ax.set_ylabel('AP')
    ax.set_title('验证集检测精度 (COCO bbox)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(plots_dir, 'efficiency_map.png')
    fig2.savefig(p2, dpi=150)
    plt_m.close(fig2)

    # --- 图2b: 尺度 AP ---
    p2b = None
    p2b_path = os.path.join(plots_dir, 'efficiency_map_scale.png')
    if any(v is not None for v in aps_list + apm_list + apl_list):
        fig2b, axb = plt_m.subplots(figsize=(8, 4.5))
        if any(v is not None for v in aps_list):
            axb.plot(x, aps_list, '-o', markersize=3, label='AP small', color='#e67e22')
        if any(v is not None for v in apm_list):
            axb.plot(x, apm_list, '-s', markersize=3, label='AP medium', color='#27ae60')
        if any(v is not None for v in apl_list):
            axb.plot(x, apl_list, '-^', markersize=3, label='AP large', color='#8e44ad')
        axb.set_xlabel('Epoch（log 记录序号）')
        axb.set_ylabel('AP')
        axb.set_title('验证集按尺度 AP（COCO）')
        axb.legend()
        axb.grid(True, alpha=0.3)
        fig2b.tight_layout()
        fig2b.savefig(p2b_path, dpi=150)
        plt_m.close(fig2b)
        p2b = p2b_path

    # --- 图3: 单 epoch  wall time ---
    eids, secs = checkpoint_epoch_seconds(result_dir)
    fig3, ax = plt_m.subplots(figsize=(8, 4.5))
    if secs:
        ax.bar([str(e) for e in eids], secs, color='steelblue', edgecolor='navy', alpha=0.85)
        ax.set_xlabel('Epoch（由相邻 checkpoint 时间差估计）')
        ax.set_ylabel('耗时 (秒)')
        ax.set_title('各 epoch 训练+验证 wall time（近似）')
        mean_s = float(np.mean(secs))
        ax.axhline(mean_s, color='orange', linestyle='--', label=f'平均 {mean_s:.1f}s')
        ax.legend()
    else:
        ax.text(0.5, 0.5, '未找到 checkpoint0000.pth 序列\n无法估计 epoch 耗时', ha='center', va='center')
        ax.set_axis_off()
    fig3.tight_layout()
    p3 = os.path.join(plots_dir, 'efficiency_epoch_time.png')
    fig3.savefig(p3, dpi=150)
    plt_m.close(fig3)

    p4 = p5 = rep_path = None
    nparam = None
    if not args.curves_only:
        # --- 图4: 汇总信息文本 ---
        info_path = os.path.join(result_dir, 'info.txt')
        if os.path.isfile(info_path):
            with open(info_path, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'number of params' in line.lower():
                        try:
                            nparam = line.split(':')[-1].strip()
                        except Exception:
                            pass
                        break

        fig4, ax = plt_m.subplots(figsize=(8, 3))
        ax.set_axis_off()
        lines = [
            'Efficiency / training summary (auto)',
            f'Epochs: {len(rows)}',
            f'Output dir: {result_dir}',
        ]
        if secs:
            lines.append(f'Avg epoch wall time (from checkpoints): {np.mean(secs):.1f} s')
            lines.append(f'Total est. train time ({len(secs)} intervals): {np.sum(secs)/60:.1f} min')
        if nparam:
            lines.append(f'Params (from info.txt): {nparam}')
        lines.append('Plots dir: ' + plots_dir)
        lines.append('Note: consecutive checkpoint epoch indices only (train+val+IO).')
        ax.text(0.02, 0.95, '\n'.join(lines), transform=ax.transAxes, va='top', fontsize=10, family='sans-serif')
        p4 = os.path.join(plots_dir, 'efficiency_summary.png')
        fig4.savefig(p4, dpi=150, bbox_inches='tight')
        plt_m.close(fig4)

        # --- 图5: 综合 dashboard ---
        fig5 = plt_m.figure(figsize=(11, 8))
        g = fig5.add_gridspec(2, 2, hspace=0.3, wspace=0.28)
        ax_a = fig5.add_subplot(g[0, 0])
        ax_a.plot(x, train_loss, 'b-o', markersize=2, label='train')
        if any(v is not None for v in test_loss):
            ax_a.plot(x, test_loss, 'r-s', markersize=2, label='val')
        ax_a.set_title('Loss')
        ax_a.set_xlabel('Epoch')
        ax_a.grid(True, alpha=0.3)
        ax_a.legend(fontsize=8)

        ax_b = fig5.add_subplot(g[0, 1])
        if any(v is not None for v in map50):
            ax_b.plot(x, map50, 'm-o', markersize=3, label='mAP@0.5')
        if any(v is not None for v in map5095):
            ax_b.plot(x, map5095, 'c-s', markersize=3, label='mAP@[.5:.95]')
        ax_b.set_title('COCO AP')
        ax_b.set_xlabel('Epoch')
        ax_b.legend(fontsize=8)
        ax_b.grid(True, alpha=0.3)

        ax_c = fig5.add_subplot(g[1, 0])
        ax_c.plot(x, lr, 'g-', linewidth=2)
        ax_c.set_title('Learning rate')
        ax_c.set_xlabel('Epoch')
        ax_c.grid(True, alpha=0.3)

        ax_d = fig5.add_subplot(g[1, 1])
        if secs:
            ax_d.bar(range(len(secs)), secs, color='teal', alpha=0.8)
            ax_d.set_xticks(range(len(secs)))
            ax_d.set_xticklabels([str(e) for e in eids], rotation=45, ha='right', fontsize=7)
            ax_d.set_title('Epoch 耗时 (s)')
        else:
            ax_d.text(0.5, 0.5, '无 checkpoint 序列', ha='center')
            ax_d.set_axis_off()
        fig5.suptitle('DINO 训练效率分析总览', fontsize=14, y=1.02)
        p5 = os.path.join(plots_dir, 'efficiency_dashboard.png')
        fig5.savefig(p5, dpi=150, bbox_inches='tight')
        plt_m.close(fig5)

        rep_path = os.path.join(plots_dir, 'analysis_report.txt')
        _write_text_report(
            rep_path, result_dir, rows, map50, map5095,
            aps_list, apm_list, apl_list, secs, nparam,
        )

    if args.curves_only:
        print(plots_dir)
    else:
        print('已保存至:', plots_dir)
        for p in (p1, p2, p2b, p3, p4, p5, rep_path):
            if p:
                print(' ', p)


if __name__ == '__main__':
    main()
