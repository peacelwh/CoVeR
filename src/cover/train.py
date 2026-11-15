"""Training and evaluation of LLM-Fusion and CoVeR (Secs. 4.1 and 4.3 of the paper).

LLM-Fusion is the backbone trained with L_L alone, i.e. CoVeR without judges and gate. CoVeR keeps the architecture,
the inputs and the inference procedure unchanged and modifies only how the model is trained: each sample is expanded
into its full and single-modality views, processed jointly and with gradients; the views are trained as unimodal
judges (Eq. 5) and their disagreement decides which samples supervise the fused path (Eqs. 3, 4, 6).

    python scripts/train.py --dataset mosi --method cover --seed 42
    python scripts/train.py --dataset mosi --method llm_fusion --seed 42
    python scripts/train.py --dataset sims --method cover --recipe default --seed 42      # tau = 0.3, lambda_uni = 1

One checkpoint per run (best validation Avg Corr) serves every metric. A run directory holds ``config.json``,
``history.json``, ``best.pt`` (trainable tensors) and ``results.json``.
"""
import argparse
import json
import os
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from cover import mrbench_a
from cover.data import MODALITIES, PERMANENT, SentimentDataset, collate, pattern_name, to_device, worker_init_fn
from cover.evaluate import (evaluate_mrbench_a, evaluate_patterns, evaluate_permanent, evaluate_rates, summarize)
from cover.learned_gate import GateObjective, LearnedGateFusion
from cover.losses import FusionObjective
from cover.model import LLMFusion
from cover.recipes import DATASETS, HUMAN_UNIMODAL, get_recipe

OVERRIDABLE = ("tau", "lambda_uni", "lora_r", "lr", "epochs", "ema", "regression", "batch", "llm_path", "num_workers")


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--method", default="cover", choices=["llm_fusion", "cover", "learned_gate"],
                    help="learned_gate: LLM-Fusion with a learned reliability gate (comparison of corrections)")
    ap.add_argument("--recipe", default="selected", choices=["selected", "default"],
                    help="selected: per-dataset hyper-parameters chosen on validation; default: default recipe of the "
                         "backbone (tau = 0.3, lambda_uni = 1 for CoVeR), used by the analyses")
    ap.add_argument("--seed", type=int, default=42)
    # hyper-parameters (default: the recipe)
    ap.add_argument("--tau", type=float, default=None, help="margin of the disagreement gate")
    ap.add_argument("--lambda_uni", type=float, default=None, help="weight of the unimodal-judge loss")
    ap.add_argument("--lora_r", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ema", type=float, default=None, help="decay of the exponential moving average of the weights (0: off)")
    ap.add_argument("--regression", default=None, choices=["mse", "l1"])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--llm_path", default=None, help="decoder-only LLM (default: Qwen/Qwen3-0.6B)")
    # variants
    ap.add_argument("--no_gate", action="store_true",
                    help="CoVeR without the disagreement gate: unimodal judges only (unimodal multi-task)")
    ap.add_argument("--conflict_weight", type=float, default=1.0,
                    help="> 1: weight of the task loss on the conflict cases of the training split (conflict re-weighting)")
    ap.add_argument("--gate_warmup", type=float, default=3.0, help="epochs over which the gate is ramped in")
    ap.add_argument("--missing", default="dynamic", choices=["dynamic", "permanent", "permanent_incomplete"],
                    help="dynamic: a pattern is drawn at every visit (main protocol); permanent[_incomplete]: every "
                         "training and validation sample keeps one pattern for the whole run")
    ap.add_argument("--mask_seed", type=int, default=0, help="permanent missingness: seed of the pattern assignment")
    # evaluation
    ap.add_argument("--eval_only", action="store_true", help="evaluate --resume (or the run's best.pt) without training")
    ap.add_argument("--resume", default=None, help="checkpoint to load")
    ap.add_argument("--tau_test", type=float, default=None, help="test-time margin of robust mode (default: tau)")
    ap.add_argument("--no_rates", action="store_true", help="skip the random-missing-rate evaluation")
    # paths
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--mrbench_dir", default="data/mrbench_a", help="MR-Bench-A annotations (<dataset>_<split>.jsonl)")
    ap.add_argument("--judge_labels_dir", default=None,
                    help="directory with the expert judgments of the training split (default: --mrbench_dir); under "
                         "permanent missingness, the output of scripts/build_mrbench_a.py --missing permanent")
    ap.add_argument("--out_root", default="results/runs")
    ap.add_argument("--name", default=None, help="run name (default: <method>[_<recipe>])")
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=None)
    return ap


def resolve_config(args) -> SimpleNamespace:
    """Recipe of the paper, then the command-line overrides."""
    cfg = get_recipe(args.method, args.dataset, args.recipe)
    for k in OVERRIDABLE:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.method != "cover":
        cfg.update(tau=None, lambda_uni=0.0)
    if args.no_gate:
        cfg["tau"] = None
    for k in ("seed", "conflict_weight", "gate_warmup", "missing", "mask_seed", "data_dir", "mrbench_dir", "eval_batch"):
        cfg[k] = getattr(args, k)
    cfg["judge_labels_dir"] = args.judge_labels_dir or args.mrbench_dir
    cfg["name"] = args.name or (args.method + ("" if args.recipe == "selected" or args.method == "learned_gate" else "_default")
                                + ("_nogate" if args.no_gate else ""))
    return SimpleNamespace(**cfg)


def setup_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, meta, device) -> LLMFusion:
    kwargs = dict(llm_path=cfg.llm_path, dims=meta["dims"], T=meta["T"], num_fusion_tokens=cfg.num_fusion_tokens,
                  lora_r=cfg.lora_r, lora_dropout=cfg.lora_dropout, dropout=cfg.dropout, grad_checkpointing=True)
    if cfg.method == "learned_gate":
        return LearnedGateFusion(gate_alpha=cfg.gate_alpha, **kwargs).to(device)
    return LLMFusion(**kwargs).to(device)


def trainable_state_dict(model):
    """The tensors a checkpoint stores: everything outside the frozen LLM, and its LoRA weights."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in trainable or not k.startswith("llm.")}


def load_checkpoint(model, path, device):
    """Strict loading of a checkpoint written by :func:`trainable_state_dict`: the architecture (LLM, number of fusion
    tokens, LoRA rank) must be the one the checkpoint was trained with."""
    state = torch.load(path, map_location=device, weights_only=True)
    own = model.state_dict()
    expected = set(trainable_state_dict(model))
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - set(own))
    mismatched = sorted(k for k in state if k in own and own[k].shape != state[k].shape)
    if missing or unexpected or mismatched:
        raise RuntimeError(f"checkpoint {path} does not match the model: missing {missing[:3]} ({len(missing)}), "
                           f"unexpected {unexpected[:3]} ({len(unexpected)}), shape mismatch {mismatched[:3]} "
                           f"({len(mismatched)}); build the model with the configuration of the run")
    model.load_state_dict(state, strict=False)
    return model


def attach_annotations(cfg, train_ds):
    """Unimodal labels of the judges and, optionally, conflict weights for the training split.

    The labels y^(j) are the human unimodal annotations on CH-SIMS / CH-SIMSv2 (stored with the corpus) and the
    out-of-fold expert predictions of MR-Bench-A on the other corpora.
    """
    need_labels = cfg.lambda_uni > 0 and cfg.dataset not in HUMAN_UNIMODAL
    if not (need_labels or cfg.conflict_weight != 1.0):
        return
    if need_labels:
        rows = mrbench_a.load(cfg.judge_labels_dir, cfg.dataset, "train", train_ds.ids)
        train_ds.set_unimodal_labels(mrbench_a.unimodal_labels(rows))
        print(f"[data] unimodal labels: out-of-fold expert predictions ({mrbench_a.path_of(cfg.judge_labels_dir, cfg.dataset, 'train')})")
    if cfg.conflict_weight != 1.0:
        rows = mrbench_a.load(cfg.mrbench_dir, cfg.dataset, "train", train_ds.ids)
        w = torch.tensor([cfg.conflict_weight if r["case_type"] == "conflict" else 1.0 for r in rows])
        train_ds.sample_weight = w
        print(f"[data] conflict re-weighting x{cfg.conflict_weight} on {int((w > 1).sum())} training samples")


def train(cfg, model, train_ds, valid_ds, device, out_dir):
    """Optimise the objective and keep the checkpoint with the best validation Avg Corr."""
    use_views = cfg.lambda_uni > 0 or cfg.tau is not None
    weights = dict(lambda_task=cfg.lambda_task, lambda_con=cfg.lambda_con, lambda_reg=cfg.lambda_reg,
                   lambda_uni=cfg.lambda_uni, regression=cfg.regression)
    if cfg.method == "learned_gate":
        objective = GateObjective(targets=cfg.gate_targets, agreement=cfg.gate_agreement, **weights)
    else:
        objective = FusionObjective(**weights)
    ramp_gate = (cfg.tau is not None or cfg.method == "learned_gate") and cfg.gate_warmup > 0
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    other = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    optim = torch.optim.AdamW([{"params": lora, "lr": cfg.lr}, {"params": other, "lr": cfg.lr}],
                              weight_decay=cfg.weight_decay)
    steps_per_epoch = (len(train_ds) + cfg.batch - 1) // cfg.batch
    total_steps, warm = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):  # linear warm-up, then cosine decay to lr_floor * lr
        if step < warm:
            return (step + 1) / max(warm, 1)
        progress = (step - warm) / max(total_steps - warm, 1)
        return cfg.lr_floor + (1 - cfg.lr_floor) * 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate,
                        drop_last=True, worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None)

    ema, ema_steps = None, 0
    if cfg.ema > 0:  # exponential moving average of the trainable weights; validation and testing use the average
        ema = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def swap_ema():
        """Exchange the live and the averaged weights in place (call twice to restore)."""
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema:
                    tmp = p.detach().clone()
                    p.copy_(ema[n])
                    ema[n].copy_(tmp)

    permanent = cfg.missing in PERMANENT
    val_mode = "robust" if cfg.tau is not None else "accuracy"  # validation scores the prediction the task loss is on
    best, best_epoch, history = -1e9, -1, []
    ckpt = os.path.join(out_dir, "best.pt")
    for epoch in range(cfg.epochs):
        model.train()
        train_ds.set_epoch(epoch)
        sums, n_seen, t0 = Counter(), 0, time.time()
        for it, batch in enumerate(loader):
            batch = to_device(batch, device)
            gate_scale = 1.0
            if ramp_gate:
                gate_scale = min(1.0, (epoch * steps_per_epoch + it) / (cfg.gate_warmup * steps_per_epoch))
            feats = {m: batch[m] for m in MODALITIES}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(feats, batch["mask"], views="judges" if use_views else None, tau=cfg.tau,
                            gate_scale=gate_scale)
                loss, parts = objective(out, batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            if ema is not None:
                ema_steps += 1
                decay = min(cfg.ema, (1 + ema_steps) / (10 + ema_steps))
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in ema:
                            ema[n].mul_(decay).add_(p.detach(), alpha=1 - decay)
            bs = batch["y"].shape[0]
            n_seen += bs
            for k, v in parts.items():
                sums[k] += v * bs
            if it % 50 == 0:
                print(f"  epoch {epoch} step {it}/{steps_per_epoch} " + " ".join(f"{k}={v / n_seen:.4f}" for k, v in sums.items())
                      + f" lr={sched.get_last_lr()[0]:.2e}", flush=True)
        row = {"epoch": epoch, "train": {k: v / max(n_seen, 1) for k, v in sums.items()}, "train_time": time.time() - t0}

        if ema is not None:
            swap_ema()
        if permanent:  # validation samples keep their own permanent pattern
            val = evaluate_permanent(model, valid_ds, cfg.eval_batch, device, val_mode, cfg.tau, cfg.num_workers)
        else:
            val = evaluate_patterns(model, valid_ds, cfg.eval_batch, device, val_mode, cfg.tau, num_workers=cfg.num_workers)
        row["valid"] = val
        score = val["Avg"]["Corr"]
        print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in row["train"].items())
              + f" | valid Avg Corr={score:.4f} MAE={val['Avg']['MAE']:.4f} Acc2={val['Avg']['Acc2']:.4f}"
              + f" | {row['train_time']:.0f}s", flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save(trainable_state_dict(model), ckpt)
            print(f"  * best validation Avg Corr {best:.4f}: checkpoint saved", flush=True)
        if ema is not None:
            swap_ema()
        history.append(row)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
    print(f"[train] done: best epoch {best_epoch}, validation Avg Corr {best:.4f}")
    return ckpt


def test(cfg, model, test_ds, device, tau_test=None, rates=True):
    """Test-time evaluation of one checkpoint: accuracy mode and, for a model with views, robust mode."""
    rows = None
    if os.path.exists(mrbench_a.path_of(cfg.mrbench_dir, cfg.dataset, "test")):
        rows = mrbench_a.load(cfg.mrbench_dir, cfg.dataset, "test", test_ds.ids)
    tau_test = tau_test if tau_test is not None else cfg.tau
    final = {}
    for mode in ("accuracy", "robust"):
        if mode == "robust" and tau_test is None:
            continue
        tau = tau_test if mode == "robust" else None
        res = {"patterns": evaluate_patterns(model, test_ds, cfg.eval_batch, device, mode, tau, num_workers=cfg.num_workers)}
        if mode == "robust":
            res["tau_test"] = tau
        if rates:
            res["rates"] = evaluate_rates(model, test_ds, cfg.eval_batch, device, mode, tau, num_workers=cfg.num_workers)
        if rows is not None:
            res["mrbench_a"] = evaluate_mrbench_a(model, test_ds, rows, cfg.eval_batch, device, mode, tau, cfg.num_workers)
        final[mode] = res
        print(f"[test] {mode} mode" + (f" (tau_test = {tau})" if tau is not None else "") + "\n" + summarize(res["patterns"]))
        if rates:
            print("  " + summarize({"Rate": res["rates"]["Rate"]}).strip())
        if rows is not None:
            m = res["mrbench_a"]
            print(f"  MR-Bench-A: Cons={100 * m['Cons']:.1f} Conf={100 * m['Conf']:.1f} ShortErr={100 * m['ShortErr']:.1f}"
                  f" ({m['n_consensus']} consensus / {m['n_conflict']} conflict cases)")
    return final


def evaluate_run(run_dir, tau_test=None, rates=True, checkpoint=None, data_dir=None, mrbench_dir=None, llm_path=None):
    """Re-evaluate a finished run from its ``config.json`` and ``best.pt``."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = SimpleNamespace(**json.load(f))
    for k, v in (("data_dir", data_dir), ("mrbench_dir", mrbench_dir), ("llm_path", llm_path)):
        if v is not None:
            setattr(cfg, k, v)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = SentimentDataset(os.path.join(cfg.data_dir, f"{cfg.dataset}.pt"), "test")
    model = build_model(cfg, test_ds.meta, device)
    load_checkpoint(model, checkpoint or os.path.join(run_dir, "best.pt"), device)
    return cfg, test(cfg, model, test_ds, device, tau_test, rates)


def main(argv=None):
    args = get_parser().parse_args(argv)
    cfg = resolve_config(args)
    setup_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_root, cfg.dataset, f"{cfg.name}_s{cfg.seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[run] {out_dir}\n[config] {vars(cfg)}")

    path = os.path.join(cfg.data_dir, f"{cfg.dataset}.pt")
    permanent = cfg.missing in PERMANENT
    train_ds = SentimentDataset(path, "train", missing=cfg.missing, seed=cfg.seed, mask_seed=cfg.mask_seed)
    valid_ds = SentimentDataset(path, "valid", missing=cfg.missing if permanent else "none", mask_seed=cfg.mask_seed)
    test_ds = SentimentDataset(path, "test")
    meta = train_ds.meta
    print(f"[data] {cfg.dataset}: train {len(train_ds)} / valid {len(valid_ds)} / test {len(test_ds)}, dims {meta['dims']}")
    if permanent:
        print(f"[data] permanent missingness ({cfg.missing}, mask_seed {cfg.mask_seed}): train "
              f"{dict(Counter(pattern_name(m) for m in train_ds.obs.tolist()))}")

    model = build_model(cfg, meta, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {cfg.llm_path}: {n_train / 1e6:.2f}M trainable of {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    ckpt = args.resume or os.path.join(out_dir, "best.pt")
    if not args.eval_only:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(vars(cfg), f, indent=1)
        attach_annotations(cfg, train_ds)
        if args.resume:
            load_checkpoint(model, args.resume, device)
        ckpt = train(cfg, model, train_ds, valid_ds, device, out_dir)
    load_checkpoint(model, ckpt, device)
    print(f"[test] checkpoint {ckpt}")
    final = test(cfg, model, test_ds, device, args.tau_test, rates=not args.no_rates)
    dst = os.path.join(out_dir, "results.json")
    with open(dst, "w") as f:
        json.dump(final, f, indent=1)
    print(f"[done] {dst}")
    return final


if __name__ == "__main__":
    main()
