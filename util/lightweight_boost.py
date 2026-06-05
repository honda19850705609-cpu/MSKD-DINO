import copy
import gc
import time

import torch
import torch.nn.functional as F


def log_gpu(tag=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        t = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU {tag}] 使用: {a:.1f}GB / 预留: {r:.1f}GB / 总计: {t:.1f}GB ({r / t * 100:.0f}%)")


def auto_adapt_gpu(args):
    if not torch.cuda.is_available():
        print("[GPU自适应] 无 GPU，batch_size=1, 无梯度累积")
        args.batch_size = 1
        args.grad_accum_steps = 1
        args.gradient_accumulation_steps = 1
        return

    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    has_distill = getattr(args, "distill_teacher_ckpt", None) not in (None, "", "None")

    if has_distill:
        mem_table = [(80, 8), (48, 6), (40, 4), (24, 3), (16, 2), (12, 1), (0, 1)]
    else:
        mem_table = [(80, 16), (48, 10), (40, 8), (24, 6), (16, 4), (12, 3), (8, 2), (0, 1)]

    safe_batch = 1
    for mem_threshold, bs in mem_table:
        if total_mem_gb >= mem_threshold:
            safe_batch = bs
            break

    target_effective = int(getattr(args, "target_effective_batch", 8) or 8)
    grad_accum = max(1, target_effective // safe_batch)
    effective = safe_batch * grad_accum
    args.batch_size = safe_batch
    args.grad_accum_steps = grad_accum
    args.gradient_accumulation_steps = grad_accum

    print(f"\n{'=' * 60}")
    print("  GPU 自适应配置")
    print(f"  GPU: {gpu_name}")
    print(f"  显存: {total_mem_gb:.1f} GB")
    print(f"  蒸馏: {'是 (teacher+student双模型)' if has_distill else '否'}")
    print(f"  batch_size: {safe_batch}")
    print(f"  梯度累积: {grad_accum} 步")
    print(f"  等效 batch: {effective}")
    print(f"{'=' * 60}\n")

    if has_distill and total_mem_gb < 14:
        print(f"[GPU自适应] 显存 {total_mem_gb:.0f}GB < 14GB，自动禁用蒸馏")
        args.distill_teacher_ckpt = None
        args.distill_weight = 0.0
        for mem_threshold, bs in [(80, 16), (48, 10), (40, 8), (24, 6), (16, 4), (12, 3), (8, 2), (0, 1)]:
            if total_mem_gb >= mem_threshold:
                safe_batch = bs
                break
        args.batch_size = safe_batch
        grad_accum = max(1, target_effective // safe_batch)
        args.grad_accum_steps = grad_accum
        args.gradient_accumulation_steps = grad_accum


def distill_kl_loss(student_logits, teacher_logits, temperature=4.0):
    t_nq = teacher_logits.shape[1]
    s_nq = student_logits.shape[1]
    if s_nq != t_nq:
        with torch.no_grad():
            s_conf = student_logits.detach().sigmoid().max(dim=-1).values
            _, topk_idx = s_conf.topk(t_nq, dim=1)
            topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, student_logits.shape[-1])
        student_logits = student_logits.gather(1, topk_idx_expanded)

    s = student_logits.reshape(-1, student_logits.shape[-1])
    t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    s_soft = F.log_softmax(s / temperature, dim=-1)
    t_soft = F.softmax(t / temperature, dim=-1)
    return F.kl_div(s_soft, t_soft, reduction="batchmean") * (temperature**2)


def build_teacher_model(args, device, build_model_main):
    teacher_args = copy.deepcopy(args)
    teacher_args.enc_layers = 6
    teacher_args.dec_layers = 6
    teacher_args.dim_feedforward = 2048
    teacher_args.bifpn_repeats = 2
    teacher_args.num_queries = 900
    teacher_args.num_select = 300
    teacher_args.dn_number = 100

    teacher_model, _, _ = build_model_main(teacher_args)
    ckpt = torch.load(args.distill_teacher_ckpt, map_location="cpu")
    if "ema_model" in ckpt:
        teacher_model.load_state_dict(ckpt["ema_model"], strict=True)
        print("[蒸馏] Teacher 加载 EMA 权重")
    elif "model" in ckpt:
        teacher_model.load_state_dict(ckpt["model"], strict=True)
        print("[蒸馏] Teacher 加载普通权重")
    else:
        teacher_model.load_state_dict(ckpt, strict=True)
        print("[蒸馏] Teacher 加载纯 state_dict 权重")

    teacher_model.to(device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    log_gpu("Teacher加载完成")
    if torch.cuda.is_available():
        remaining_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        if remaining_gb < 3.0:
            print(f"[蒸馏] 警告：Teacher加载后仅剩 {remaining_gb:.1f}GB 显存，自动禁用蒸馏")
            del teacher_model
            gc.collect()
            torch.cuda.empty_cache()
            return None
    return teacher_model


def setup_distillation(args, device, build_model_main):
    if not getattr(args, "distill_teacher_ckpt", None):
        print("[蒸馏] 未指定 teacher，跳过")
        return None
    try:
        teacher = build_teacher_model(args, device, build_model_main)
        if teacher is not None:
            print("[蒸馏] Teacher 就绪")
        else:
            print("[蒸馏] 显存不足，退回纯轻量化训练")
        return teacher
    except Exception as e:
        print(f"[蒸馏] 加载失败: {e}，退回纯轻量化训练")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def efficiency_report(model, device):
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 60}")
    print("  效率评测")
    print(f"  参数量: {total / 1e6:.2f}M")
    for prefix in ["backbone", "transformer.encoder", "transformer.decoder", "bifpn", "input_proj"]:
        cnt = sum(p.numel() for n, p in model.named_parameters() if prefix in n)
        if cnt > 0:
            print(f"    {prefix}: {cnt / 1e6:.2f}M")
    if device.type == "cuda":
        dummy = torch.randn(1, 3, 800, 1333).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            for _ in range(50):
                model(dummy)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(200):
                model(dummy)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
        fps = 200 / elapsed
        print(f"  FPS: {fps:.1f}")
        print(f"  延迟: {elapsed / 200 * 1000:.1f} ms")
    else:
        fps = 0.0
        print("  FPS: N/A (CPU)")
    print(f"{'=' * 60}\n")
    return {"params_M": total / 1e6, "fps": fps}
