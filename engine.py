# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch
import numpy as np

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.lightweight_boost import distill_kl_loss

_ZERO_AP_HINT_SHOWN = False


def _maybe_log_coco_bbox_near_zero_ap(coco_eval_bbox, logger):
    """训练初中期 mAP@0.5 长期为 0 时给一次性排查提示（不刷屏）。"""
    global _ZERO_AP_HINT_SHOWN
    if _ZERO_AP_HINT_SHOWN or coco_eval_bbox is None:
        return
    st = getattr(coco_eval_bbox, 'stats', None)
    if st is None or len(st) < 2:
        return
    if float(st[1]) >= 1e-4:
        return
    _ZERO_AP_HINT_SHOWN = True
    log = logger.warning if logger else print
    log(
        '[coco_eval] 验证 bbox mAP@0.5 接近 0（stats[1]）。注意: stats[0] 为 mAP@[.5:.95]，'
        'VisDrone 小目标多时早期 epoch 上 [0] 常比 [1] 更「难看」，但 [1] 也应缓慢上升。'
    )
    log(
        '[coco_eval] 排查: (1) 是否加载 COCO 预训练 --pretrain_model_path（无预训练时前几 epoch AP 常接近 0）；'
        '(2) --resume_allow_partial 是否导致检测头随机初始化；'
        '(3) 类别 id 与 num_classes=10；'
        '(4) 设 DINO_EVAL_DEBUG=1 看首张图 max(score)；'
        '(5) tools/visdrone_draw_predictions.py 可视化。'
    )


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,
                    model_without_ddp=None, teacher_model=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    accum_steps = int(getattr(args, 'gradient_accumulation_steps', 1) or 1) if args is not None else 1
    if accum_steps < 1:
        accum_steps = 1

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad(set_to_none=True)
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            if teacher_model is not None and float(getattr(args, 'distill_weight', 0.0) or 0.0) > 0:
                with torch.no_grad():
                    t_out = teacher_model(samples)
                d_loss = distill_kl_loss(
                    outputs['pred_logits'],
                    t_out['pred_logits'],
                    temperature=float(getattr(args, 'distill_temperature', 4.0) or 4.0),
                )
                losses = losses + float(args.distill_weight) * d_loss
                loss_dict['loss_distill'] = d_loss

        # 梯度累积：保持等效 batch 的数学等价（loss 按累积步数缩放）
        if accum_steps > 1:
            losses = losses / float(accum_steps)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            scaler.scale(losses).backward()
            do_step = ((_cnt + 1) % accum_steps == 0)
            if do_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            # original backward function
            losses.backward()
            do_step = ((_cnt + 1) % accum_steps == 0)
            if do_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Scheduler / EMA should follow optimizer steps (not every micro-batch)
        if do_step and args.onecyclelr:
            lr_scheduler.step()
        if do_step and args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        _empty = int(getattr(args, 'cuda_empty_cache_every_iters', 0) or 0)
        if _empty > 0 and device.type == 'cuda' and _cnt % _empty == 0:
            torch.cuda.empty_cache()
        _every = int(getattr(args, 'checkpoint_every_iters', 0) or 0)
        if (
            _every > 0
            and (args.output_dir or '').strip()
            and model_without_ddp is not None
            and _cnt % _every == 0
        ):
            from pathlib import Path
            od = Path(args.output_dir)
            weights = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'intra_epoch_checkpoint': True,
            }
            if lr_scheduler is not None:
                weights['lr_scheduler'] = lr_scheduler.state_dict()
            if getattr(args, 'use_ema', False) and ema_m is not None:
                weights['ema_model'] = ema_m.module.state_dict()
            emerg = od / 'checkpoint_emergency.pth'
            tmp = od / 'checkpoint_emergency.pth.part'
            utils.save_on_master(weights, tmp)
            if utils.is_main_process() and tmp.is_file():
                tmp.replace(emerg)
            if logger and utils.is_main_process():
                logger.info(
                    'Saved emergency checkpoint (epoch %s, iter %s in epoch): %s',
                    epoch, _cnt, emerg,
                )
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    # epoch 结束：若最后不足 accum_steps 仍有梯度未 step，做一次收尾 step（不丢梯度）
    if accum_steps > 1 and (_cnt % accum_steps != 0):
        if args.amp:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema and epoch >= args.ema_epoch:
            ema_m.update(model)

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    _eval_debug_done = False
    output_state_dict = {} # for debug only
    _pred_by_image = {}  # for P/R/F1 (single GPU only)
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            # 验证只做纯推理：勿传 targets，避免与 train 共用 DN 分支时产生歧义（eval 下 DN 本不生效）
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if (
            os.environ.get("DINO_EVAL_DEBUG", "").strip().lower() in ("1", "true", "yes", "y")
            and utils.is_main_process()
            and not _eval_debug_done
        ):
            _eval_debug_done = True
            dbg = print if logger is None else logger.info
            for ti, r in enumerate(results):
                sc = r.get("scores")
                bx = r.get("boxes")
                if sc is None or sc.numel() == 0:
                    dbg("[DINO_EVAL_DEBUG] batch0 pred: empty scores")
                    break
                dbg(
                    "[DINO_EVAL_DEBUG] sample %d: pred max(score)=%.5f mean(top10)=%.5f "
                    "boxes xyxy min/max=%s/%s orig_size=%s",
                    ti,
                    float(sc.max()),
                    float(sc[:10].mean()) if sc.numel() >= 10 else float(sc.mean()),
                    bx.min().tolist() if bx is not None else None,
                    bx.max().tolist() if bx is not None else None,
                    orig_target_sizes[ti].tolist(),
                )
                break
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)
        # collect raw predictions for PRF1 (rank0 single-GPU only; distributed will skip later)
        if args is not None and (not getattr(args, "distributed", False)) and utils.is_main_process():
            for image_id, out in res.items():
                try:
                    _pred_by_image[int(image_id)] = {
                        "boxes": out["boxes"].detach().cpu().numpy().astype(np.float32),
                        "labels": out["labels"].detach().cpu().numpy().astype(np.int64),
                        "scores": out["scores"].detach().cpu().numpy().astype(np.float32),
                    }
                except Exception:
                    pass

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        if 'bbox' in coco_evaluator.coco_eval:
            _maybe_log_coco_bbox_near_zero_ap(coco_evaluator.coco_eval['bbox'], logger)
        if getattr(args, 'log_thesis_metrics', False) and 'bbox' in coco_evaluator.coco_eval:
            try:
                from util.thesis_metrics import log_thesis_coco_bbox_metrics
                log_thesis_coco_bbox_metrics(coco_evaluator.coco_eval['bbox'], logger)
            except Exception as ex:
                print(f'[thesis_metrics] skipped: {ex}')
        # Phase-A: log Precision/Recall/F1 (best threshold scan) for VisDrone
        if getattr(args, 'log_thesis_metrics', False) and utils.is_main_process():
            if getattr(args, "distributed", False):
                (logger.info if logger else print)(
                    "[任务书说明] Precision/Recall/F1 统计在分布式下默认跳过（需额外 gather）。"
                )
            else:
                try:
                    from util.prf1_metrics import find_best_f1_threshold
                    from util.thesis_spec import CORE_CLASS_INDICES_TASKBOOK

                    # build GT dict from targets in dataset (COCO api)
                    gt_by_image = {}
                    coco = base_ds
                    img_ids = coco.getImgIds()
                    for img_id in img_ids:
                        ann_ids = coco.getAnnIds(imgIds=[img_id])
                        anns = coco.loadAnns(ann_ids)
                        if not anns:
                            gt_by_image[int(img_id)] = {
                                "boxes": np.zeros((0, 4), dtype=np.float32),
                                "labels": np.zeros((0,), dtype=np.int64),
                            }
                            continue
                        boxes = []
                        labels = []
                        for a in anns:
                            if a.get("iscrowd", 0) == 1:
                                continue
                            x, y, w, h = a["bbox"]
                            boxes.append([x, y, x + w, y + h])
                            labels.append(int(a["category_id"]))
                        gt_by_image[int(img_id)] = {
                            "boxes": np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
                            "labels": np.asarray(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
                        }

                    # score threshold scan (favor recall a bit by including low thresholds)
                    thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
                    best_thr_all, best_all = find_best_f1_threshold(
                        gt_by_image=gt_by_image,
                        pred_by_image=_pred_by_image,
                        thresholds=thresholds,
                        iou_thr=0.5,
                        class_ids=None,
                    )
                    best_thr_core, best_core = find_best_f1_threshold(
                        gt_by_image=gt_by_image,
                        pred_by_image=_pred_by_image,
                        thresholds=thresholds,
                        iou_thr=0.5,
                        class_ids=CORE_CLASS_INDICES_TASKBOOK,
                    )
                    log = logger.info if logger else print
                    log(
                        "[任务书 PRF1@IoU0.5] "
                        f"All(best score_thr={best_thr_all:.2f}): "
                        f"P={best_all.precision:.4f} R={best_all.recall:.4f} F1={best_all.f1:.4f} "
                        f"(TP={best_all.tp} FP={best_all.fp} FN={best_all.fn})"
                    )
                    log(
                        "[任务书 PRF1@IoU0.5 核心类] "
                        f"Core(best score_thr={best_thr_core:.2f}): "
                        f"P={best_core.precision:.4f} R={best_core.recall:.4f} F1={best_core.f1:.4f} "
                        f"(TP={best_core.tp} FP={best_core.fp} FN={best_core.fn})"
                    )
                except Exception as ex:
                    (logger.info if logger else print)(f"[任务书 PRF1] skipped: {ex}")
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res
