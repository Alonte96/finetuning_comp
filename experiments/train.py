"""One training loop, three methods. The only branch is optimizer construction.

    python experiments/train.py --method galore --lr 1e-5
    python experiments/train.py --method lora   --lr 3e-4
    python experiments/train.py --method full   --lr 2e-5
    python experiments/train.py --method galore --smoke true   # CPU, no network

Every run appends to results/runs.jsonl: config, per-step losses, eval
loss/perplexity, peak memory (probe + analytic), tokens/sec, and -- on OOM --
a record of the failure, because on small cards full FT OOMing IS the result.
"""

from __future__ import annotations

import gc
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RunConfig, append_jsonl
from gradproj.memory import GIB, MemoryProbe, predict, spec_from_model
from gradproj.param_groups import galore_param_groups, parameter_summary


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_oom(exc: BaseException) -> bool:
    """CUDA reports exhaustion as OutOfMemoryError, but cuBLAS/cuDNN allocation
    failures inside a near-full allocator surface as plain RuntimeErrors."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in text
        or "cublas_status_alloc_failed" in text
        or "cudnn_status_alloc_failed" in text
    )


def build_autocast(cfg: RunConfig, device: torch.device):
    """Autocast context factory, with the dtype guards this benchmark needs.

    fp16 is refused rather than silently mis-trained: it needs loss scaling, and
    a GradScaler cannot unscale gradients that GaLore's layerwise hooks have
    already projected and freed during backward.
    """
    if cfg.dtype not in ("bf16", "fp16", "fp32"):
        raise SystemExit(f"--dtype must be bf16|fp16|fp32, got {cfg.dtype}")
    if cfg.dtype == "fp16":
        raise SystemExit(
            "--dtype fp16 is not supported: fp16 autocast requires loss scaling, and "
            "GradScaler cannot see gradients that layerwise projection frees during "
            "backward. Use --dtype bf16 (default) or fp32."
        )
    if cfg.dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit(
            f"--dtype bf16 needs a bf16-capable GPU; {torch.cuda.get_device_name(device)} "
            "(pre-Ampere) is not. Re-run every method with --dtype fp32 so the "
            "comparison stays fair."
        )

    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[cfg.dtype]
    if autocast_dtype is not None and device.type in ("cuda", "mps"):
        return lambda: torch.autocast(device_type=device.type, dtype=autocast_dtype)
    import contextlib

    return contextlib.nullcontext


def build_model_and_data(cfg: RunConfig, device: torch.device):
    """Real TinyLlama + Alpaca, or an offline tiny model + synthetic tokens."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

    if cfg.smoke:
        from data import load_synthetic

# improvement
        config = LlamaConfig(
            hidden_size=64, intermediate_size=176, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
            max_position_embeddings=cfg.seq_len, tie_word_embeddings=False,
        )
        model = LlamaForCausalLM(config)
        train_loader, eval_loader = load_synthetic(
            vocab=256, seq_len=cfg.seq_len, eval_examples=cfg.eval_examples,
            seed=cfg.seed, micro_batch=cfg.micro_batch,
        )
    else:
        from data import load_alpaca

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        # fp32 master weights: the fair baseline for all three methods.
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32)
        train_loader, eval_loader = load_alpaca(
            tokenizer, dataset_id=cfg.dataset_id, seq_len=cfg.seq_len,
            eval_examples=cfg.eval_examples, seed=cfg.seed, micro_batch=cfg.micro_batch,
        )

    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model.to(device), train_loader, eval_loader


def build_optimizer(cfg: RunConfig, model):
    """The ONLY method-specific code path in the whole experiment."""
    layerwise_ctx = None

    if cfg.method == "full":
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
# improvement

    elif cfg.method == "lora":
        from peft import LoraConfig, get_peft_model

        from gradproj.param_groups import DEFAULT_TARGET_MODULES

        present = {
            name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
            for name, p in model.named_parameters() if p.ndim == 2
        }
        targets = sorted(present & set(DEFAULT_TARGET_MODULES))
        peft_cfg = LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
            target_modules=targets, task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        if cfg.grad_checkpointing:
            model.enable_input_require_grads()  # needed: inputs to frozen blocks
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

    elif cfg.method == "galore":
        from gradproj.layerwise import LayerwiseProjection
        from gradproj.wrapper import ProjectedOptimizer

        groups = galore_param_groups(
            model,
            project_embeddings=cfg.galore_project_embeddings,
        )
        opt = ProjectedOptimizer(
            groups, torch.optim.AdamW,
# improvement
            rank=cfg.galore_rank, update_proj_gap=cfg.galore_update_proj_gap,
            scale=cfg.galore_scale, proj_dtype=torch.bfloat16,
            weight_decay=cfg.weight_decay, lr=cfg.lr,
        )
        if cfg.galore_layerwise:
            layerwise_ctx = LayerwiseProjection(opt).attach()

    else:
        raise ValueError(cfg.method)

    return model, opt, layerwise_ctx


@torch.no_grad()
def evaluate(model, eval_loader, device, autocast_ctx) -> dict:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
#         with autocast_ctx():
            out = model(**batch)
        n = (batch["labels"] != -100).sum().item()
        # HF averages over non-ignored tokens; re-weight to a token-level sum.
        total_nll += out.loss.item() * n
        total_tokens += n
    model.train()
    nll = total_nll / max(total_tokens, 1)
    return {"eval_loss": nll, "eval_ppl": math.exp(min(nll, 20))}


def main(argv=None):
    cfg = RunConfig.from_args(argv)
    device = pick_device()
    torch.manual_seed(cfg.seed)

    out_path = Path(cfg.out_dir) / ("smoke_runs.jsonl" if cfg.smoke else "runs.jsonl")
    run_name = cfg.resolved_run_name()
# 
    record = {
        "run": run_name,
        "phase": cfg.phase,
        "config": cfg.as_dict(),
        "fairness": cfg.fairness_fingerprint(),
        "device": str(device),
        "status": "started",
        "steps": [],
        "evals": [],
    }

    autocast_ctx = build_autocast(cfg, device)

    # An eval_every of <=0, or one larger than max_steps, means "no eval at all"
    # -- the memory bench uses it so eval activations never land in a peak-memory
    # measurement. Anything else also evaluates on the final step. Never let this
    # be silent: a run that quietly produces no quality number looks like a
    # completed run in the table.
    evaluating = 0 < cfg.eval_every <= cfg.max_steps
    if not evaluating:
        print(f"[{run_name}] eval DISABLED (--eval-every {cfg.eval_every} vs "
              f"--max-steps {cfg.max_steps}): this run reports memory and speed, "
              f"no eval loss")

    model = opt = layerwise = sched = None
    train_loader = eval_loader = data_iter = None

    probe = MemoryProbe(device)
    try:
        with probe:
            model, train_loader, eval_loader = build_model_and_data(cfg, device)
            # Spec BEFORE any peft wrapping: peft renames modules, which would
            # break target matching in the analytic prediction.
            base_spec = spec_from_model(model)
            model, opt, layerwise = build_optimizer(cfg, model)

            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, (s + 1) / max(cfg.warmup_steps, 1)),
            )

            summary = parameter_summary(model)
            record["trainable_params"] = summary["trainable"]
            record["trainable_pct"] = summary["trainable_pct"]
            record["analytic_prediction"] = _analytic(cfg, base_spec)

            print(f"[{run_name}] {device} | trainable {summary['trainable_pct']:.1f}% "
                  f"of {summary['total'] / 1e6:.1f}M params")

            model.train()
            step, tokens_seen, t0 = 0, 0, time.perf_counter()
            data_iter = iter(train_loader)
            step_times = []

            while step < cfg.max_steps:
                t_step = time.perf_counter()
                opt.zero_grad()
                for _ in range(cfg.grad_accum):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(train_loader)
                        batch = next(data_iter)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with autocast_ctx():
                        loss = model(**batch).loss
                    (loss / cfg.grad_accum).backward()
                    tokens_seen += int(batch["attention_mask"].sum())
                opt.step()
                sched.step()
                step += 1
                step_times.append(time.perf_counter() - t_step)

                if step % cfg.log_every == 0 or step == 1:
                    record["steps"].append(
                        {"step": step, "loss": float(loss.item()),
                         "lr": sched.get_last_lr()[0],
                         "step_time_s": step_times[-1]}
                    )
                    print(f"  step {step:>5} loss {loss.item():.4f} "
                          f"({step_times[-1]:.2f}s/step)")

                if evaluating and (step % cfg.eval_every == 0 or step == cfg.max_steps):
                    ev = evaluate(model, eval_loader, device, autocast_ctx)
                    ev["step"] = step
                    record["evals"].append(ev)
                    print(f"  step {step:>5} eval_loss {ev['eval_loss']:.4f} "
                          f"ppl {ev['eval_ppl']:.2f}")

            wall = time.perf_counter() - t0
            record.update(
                status="completed",
                wall_seconds=wall,
                tokens_per_sec=tokens_seen / wall,
                mean_step_time_s=sum(step_times) / len(step_times),
                # First-step SVD cost etc. shows up as max vs mean.
                max_step_time_s=max(step_times),
            )
            if hasattr(opt, "memory_breakdown"):
                record["optimizer_memory_measured"] = opt.memory_breakdown()

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not _is_oom(e):
#             raise
        record.update(status="oom", error=str(e).split("\n")[0])
        print(f"[{run_name}] OOM -- recorded as a result: {record['error']}")
    finally:
        # Layerwise hooks are held by autograd's C++ state, which Python's cyclic
        # GC cannot traverse: without this the model+optimizer stay resident for
        # the rest of the process and poison the next config's peak measurement.
        if layerwise is not None:
            layerwise.detach()
        record["memory"] = probe.report()
        append_jsonl(out_path, record)
        print(f"[{run_name}] {record['status']} | peak "
              f"{record['memory']['peak_gib']:.2f} GiB "
              f"({'exact' if record['memory']['exact'] else 'approx'}) "
              f"-> {out_path}")

# improvement
    if cfg.save_model and record["status"] == "completed":
        save_dir = Path(cfg.out_dir) / "checkpoints" / run_name
        model.save_pretrained(save_dir)

    # Drivers (bench_memory, sweep_lr) call this repeatedly in one process, so
    # every run must return the device to a clean baseline.
    model = opt = layerwise = sched = None
    train_loader = eval_loader = data_iter = None
    loss = batch = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return record


def _analytic(cfg: RunConfig, spec) -> dict:
    kwargs = {}
    if cfg.method == "lora":
        kwargs = dict(lora_rank=cfg.lora_rank)
    elif cfg.method == "galore":
        kwargs = dict(rank=cfg.galore_rank, layerwise=cfg.galore_layerwise,
                      project_embeddings=cfg.galore_project_embeddings)
    est = predict(spec, cfg.method, **kwargs)
    return {**est.as_gib(), "trainable_pct": est.trainable_pct, "notes": est.notes}
# 

if __name__ == "__main__":
    main()

# Refined

# Enhanced

# Enhanced

# Optimized

# improvement
# Optimized

# Enhanced

# Enhanced

# Optimized

# Refined

# Optimized

# Optimized

# Refined

# Refined

# Optimized

# Refined

# Refined
