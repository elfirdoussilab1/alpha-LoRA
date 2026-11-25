"""LoRA vs alpha-LoRA on GLUE (roberta-base) with IDENTICAL hyper-parameters.

Answers the reviewers' concern (AC bullet 2, LUNR (1), 4qhw W3, AI review): everything shared by the two methods
is the same -- adapter learning rate, schedule, epochs, batch size, LoRA rank/targets, seeds, data splits and
checkpoint rule -- for every task and every seed. The ONLY difference is the alpha vector of alpha-LoRA
(W = diag(alpha) W* + BA), trained as in the paper (Algorithm 1): own Adam optimizer, one step every T adapter
steps on a fresh batch; its hyper-parameters (T, lr_alpha) are also fixed for all tasks and seeds.
alpha-LoRA-theory (--method alpha-lora-theory) instead sets alpha in closed form at a few points of training, from the
mean / covariance of the margin and its per-layer alpha-derivatives on a PROBE set held out from the adapter (as in
commonsense_reasoning/train.py). Only this method loses the probe examples from its training set, unless
--probe_source dev (the probe is the checkpoint-selection split) or --probe_reinsert (it rejoins training after the last
update).
Other baselines / variants (same protocol): --method dora | ia3, --loraplus_ratio 16 (LoRA+, also combinable with alpha),
and alpha-LoRA-grad-v2 (--alpha grad --alpha_data heldout --alpha_start 0.2 --alpha_param log --alpha_accum 2).

Protocol:
  * a fixed DEV split is carved out of the GLUE train set (same for all runs) and used only to pick the checkpoint;
  * the official GLUE validation set is used only as the TEST set;
  * we report test accuracy at the best-dev checkpoint and at the last step.
Output: <out_dir>/<run_id>.json when finished (resumable queue marker), <run_id>.progress.json while running.
"""
import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import IA3Config, LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

# task -> (text keys, test split, num labels, max_len)
TASKS = {
    "rte": (("sentence1", "sentence2"), "validation", 2, 256),
    "mrpc": (("sentence1", "sentence2"), "validation", 2, 128),
    "sst2": (("sentence", None), "validation", 2, 128),
    "qnli": (("question", "sentence"), "validation", 2, 256),
    "mnli": (("premise", "hypothesis"), "validation_matched", 3, 256),
    "qqp": (("question1", "question2"), "validation", 2, 128),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FacebookAI/roberta-base")
    ap.add_argument("--task", default="rte", choices=list(TASKS))
    ap.add_argument("--method", default="lora", choices=["lora", "dora", "ia3", "alpha-lora", "alpha-lora-theory"],
                    help="alpha-lora / alpha-lora-theory are shorthands for --method lora --alpha grad / theory")
    ap.add_argument("--alpha", default="none", choices=["none", "grad", "theory"])
    ap.add_argument("--loraplus_ratio", type=float, default=0.0, help=">0: LoRA+ (lr of B = ratio * lr)")
    # shared hyper-parameters (identical for both methods)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--target_modules", default="query,value")
    ap.add_argument("--seed", type=int, default=1)
    # alpha-LoRA only (fixed for all tasks and seeds)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--lr_alpha", type=float, default=5e-3)
    ap.add_argument("--alpha_data", default="fresh", choices=["fresh", "heldout"])
    ap.add_argument("--alpha_start", type=float, default=0.0, help="fraction of training before alpha starts moving")
    ap.add_argument("--alpha_accum", type=int, default=1)
    ap.add_argument("--alpha_param", default="linear", choices=["linear", "log"])
    # alpha-LoRA-theory only
    ap.add_argument("--theory_at", default="0.3,0.6", help="fractions of training at which alpha is re-estimated")
    ap.add_argument("--theory_groups", default="global", choices=["global", "type", "layer", "module"])
    ap.add_argument("--theory_shrink", type=float, default=0.1)
    ap.add_argument("--theory_trust", type=float, default=0.25)
    ap.add_argument("--probe_frac", type=float, default=0.1)
    ap.add_argument("--probe_max", type=int, default=1000)
    ap.add_argument("--probe_source", default="train", choices=["train", "dev"],
                    help="train: probe held out of the training split; dev: reuse the checkpoint-selection split")
    ap.add_argument("--probe_reinsert", action="store_true",
                    help="probe examples rejoin the training set after the last alpha-theory update")
    # protocol / io
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--dev_max", type=int, default=2000)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--num_evals", type=int, default=20)
    ap.add_argument("--out_dir", default="glue_matched/results")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    if a.method in ("alpha-lora", "alpha-lora-theory"):
        a.method, a.alpha = "lora", ("grad" if a.method == "alpha-lora" else "theory")
    if a.alpha != "none" and a.method != "lora":
        raise ValueError("alpha rescaling is defined on top of LoRA (/LoRA+)")
    if a.alpha == "theory":
        a.alpha_param = "linear"
    if a.label is None:
        a.label = {"lora": "LoRA", "dora": "DoRA", "ia3": "IA3"}[a.method] if a.alpha == "none" else f"alpha-LoRA-{a.alpha}"
        if a.loraplus_ratio > 0:
            a.label = "LoRA+" if a.label == "LoRA" else a.label + "+LoRA+"
    a.run_id = a.run_id or f"{a.task}_{a.label.replace('+', 'plus')}_lr{a.lr}_s{a.seed}"
    return a


def uses_probe(args):   # methods that need examples held out from the adapter
    return args.alpha == "theory" or (args.alpha == "grad" and args.alpha_data == "heldout")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_data(args, tok):
    keys, test_split, num_labels, max_len = TASKS[args.task]
    raw = load_dataset("nyu-mll/glue", args.task)
    n_dev = int(min(args.dev_max, max(200, args.dev_frac * len(raw["train"]))))
    sp = raw["train"].train_test_split(test_size=n_dev, seed=args.data_seed)
    train, probe, full = sp["train"], None, None
    if uses_probe(args) and args.probe_source == "dev":     # no training example is lost
        probe = sp["test"]
    elif uses_probe(args):
        n_probe = int(min(args.probe_max, max(100, args.probe_frac * len(train))))
        pp = train.train_test_split(test_size=n_probe, seed=args.data_seed)
        train, probe = pp["train"], pp["test"]
        full = sp["train"] if args.probe_reinsert else None

    def prep(ds):
        ds = ds.map(lambda ex: tok(ex[keys[0]], ex[keys[1]] if keys[1] else None, truncation=True, max_length=max_len),
                    batched=True)
        keep = {"input_ids", "attention_mask", "label"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep]).rename_column("label", "labels")
        ds.set_format("torch")
        return ds
    opt = lambda ds: prep(ds) if ds is not None else None
    return prep(train), prep(sp["test"]), prep(raw[test_split]), opt(probe), opt(full), num_labels


class Probe:   # alpha-LoRA-theory: per-example d margin / d alpha_module, filled by the alpha hooks
    on = False
    acc = {}


def make_hook(p, name, param):
    def hook(m, i, o):
        y = o.float() * (p.exp() if param == "log" else p)
        if Probe.on and y.requires_grad:
            z = o.detach()
            y.register_hook(lambda g, z=z: Probe.acc.__setitem__(name, (g.float() * z.float()).flatten(1).sum(1)))
        return y
    return hook


def attach_alpha(model, param):
    """alpha-LoRA: one scale per output neuron of every frozen matrix adapted by LoRA (applied in fp32)."""
    named = []
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLayer) and not hasattr(mod, "alpha_scale"):
            base = mod.get_base_layer()
            p = nn.Parameter(torch.full((base.out_features,), 0.0 if param == "log" else 1.0, device=base.weight.device))
            mod.register_parameter("alpha_scale", p)
            base.register_forward_hook(make_hook(p, name, param))
            named.append((name, p))
    return named


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        correct += (logits.argmax(-1) == b["labels"]).sum().item(); total += len(b["labels"])
    model.train()
    return correct / total


# ----------------------------------------------------------------------------------------------- alpha-LoRA-theory
def phi(x):  # standard normal CDF
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def group_index(names, how):
    keys = []
    for n in names:
        m = re.search(r"layers?\.(\d+)\.", n)
        layer = int(m.group(1)) if m else -1
        typ = n.split(".")[-1]
        keys.append({"global": 0, "type": typ, "layer": layer, "module": n}[how])
    uniq = sorted(set(keys), key=str)
    return [uniq.index(k) for k in keys], len(uniq)


def theory_update(model, named_alpha, probe_loader, device, args):
    """Closed-form alpha step: maximise the predicted accuracy Phi(E[d] / std[d]) of the margin d, linearised in the
    per-group alpha shifts, with shrinkage on the covariance and a trust region on the shift."""
    names, params = [n for n, _ in named_alpha], [p for _, p in named_alpha]
    gidx, K = group_index(names, args.theory_groups)
    Gm = torch.zeros(len(names), K, device=device)
    Gm[torch.arange(len(names)), torch.tensor(gidx)] = 1.0
    saved = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    model.eval(); Probe.on = True     # roberta-base: no gradient checkpointing, eval mode only disables dropout
    Z = []
    for b in probe_loader:
        b = {k: v.to(device) for k, v in b.items()}
        Probe.acc = {}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float()
        ar = torch.arange(len(b["labels"]), device=device)
        other = logits.clone(); other[ar, b["labels"]] = -1e9
        d = logits[ar, b["labels"]] - other.max(1).values
        torch.autograd.grad(d.sum(), params)
        D = torch.stack([Probe.acc[n] for n in names], 1)            # [b, modules]
        Z.append(torch.cat([d.detach()[:, None], D @ Gm], 1).double().cpu())
    Probe.on = False
    for p in model.parameters():
        p.requires_grad_(saved[id(p)])
    model.train()
    Z = torch.cat(Z).numpy()
    mu, S = Z.mean(0), np.atleast_2d(np.cov(Z, rowvar=False))
    Ss = S + args.theory_shrink * np.diag(np.diag(S)) + 1e-12 * np.eye(len(mu))
    u = np.linalg.solve(Ss, mu)
    info = {"n_probe": len(Z), "groups": K, "pred_acc_before": phi(mu[0] / np.sqrt(S[0, 0]))}
    if u[0] <= 0:
        info.update(skipped=True)
        return info
    delta = np.clip(u[1:] / u[0], -args.theory_trust, args.theory_trust)
    v = np.concatenate([[1.0], delta])
    info.update(skipped=False, pred_acc_after=phi(mu @ v / np.sqrt(v @ S @ v)),
                delta_mean=float(delta.mean()), delta_min=float(delta.min()), delta_max=float(delta.max()),
                frac_clipped=float(np.mean(np.abs(u[1:] / u[0]) > args.theory_trust)))
    with torch.no_grad():
        for p, g in zip(params, gidx):
            p.add_(float(delta[g]))
    return info


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, f"{args.run_id}.json")
    prog_path = os.path.join(args.out_dir, f"{args.run_id}.progress.json")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    train, dev, test, probe, full, num_labels = build_data(args, tok)
    coll = DataCollatorWithPadding(tok)
    loader = lambda ds, shuf, seed=0, bs=args.batch_size: DataLoader(
        ds, batch_size=bs, shuffle=shuf, collate_fn=coll, drop_last=shuf, generator=torch.Generator().manual_seed(seed))
    train_loader = loader(train, True, args.seed)
    dev_loader, test_loader = loader(dev, False, bs=128), loader(test, False, bs=128)
    probe_loader = loader(probe, False) if probe is not None else None

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    if args.method == "ia3":   # PEFT defaults for roberta: key, value and the FFN output (output.dense)
        cfg = IA3Config(task_type=TaskType.SEQ_CLS)
    else:
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=args.lora_dropout, target_modules=args.target_modules.split(","),
                         use_dora=args.method == "dora")
    model = get_peft_model(model, cfg)
    named_alpha = attach_alpha(model, args.alpha_param) if args.alpha != "none" else []
    alpha = [p for _, p in named_alpha]
    if args.alpha == "theory":
        for p in alpha:
            p.requires_grad_(False)       # alpha only moves through theory_update
    model.to(device)
    a_ids = {id(p) for p in alpha}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and id(p) not in a_ids]  # adapter + head
    adapter = [p for _, p in named]
    if args.loraplus_ratio > 0:
        groups = [{"params": [p for n, p in named if "lora_B" not in n], "lr": args.lr},
                  {"params": [p for n, p in named if "lora_B" in n], "lr": args.lr * args.loraplus_ratio}]
    else:
        groups = [{"params": adapter, "lr": args.lr}]

    total = int(math.ceil(args.epochs * len(train_loader)))
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    opt_alpha = torch.optim.Adam(alpha, lr=args.lr_alpha, betas=(0.9, 0.99)) if args.alpha == "grad" else None
    theory_steps = ({max(1, int(float(f) * total)) for f in args.theory_at.split(",")}
                    if args.alpha == "theory" else set())

    def batches(ds):
        g = torch.Generator().manual_seed(args.seed + 7919)
        while True:
            for b in DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True, collate_fn=coll,
                                drop_last=True, generator=g):
                yield b
    alpha_it = batches(probe if args.alpha_data == "heldout" else train) if args.alpha == "grad" else None

    def loss_on(b):
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            return model(**b).loss

    n_ad, n_al = sum(p.numel() for p in adapter), sum(p.numel() for p in alpha)
    print(f"[{args.run_id}] train={len(train)} probe={len(probe) if probe is not None else 0} dev={len(dev)} "
          f"test={len(test)} steps={total} "
          f"trainable={n_ad} alpha={n_al} ({100 * n_al / n_ad:.2f}% of trainable)", flush=True)
    eval_every = max(1, total // args.num_evals)
    best, history, theory_log, step, t0 = {"dev": -1.0}, [], [], 0, time.time()
    model.train()
    while step < total:
        restart = False
        for b in train_loader:
            loss = loss_on(b)
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            for p in alpha:
                p.grad = None
            step += 1
            if opt_alpha is not None and step % args.T == 0 and step >= args.alpha_start * total:   # Algorithm 1
                gsum = [torch.zeros_like(p) for p in alpha]
                for _ in range(args.alpha_accum):
                    for gs, g in zip(gsum, torch.autograd.grad(loss_on(next(alpha_it)), alpha)):
                        gs += g / args.alpha_accum
                for p, g in zip(alpha, gsum):
                    p.grad = g
                opt_alpha.step(); opt_alpha.zero_grad(set_to_none=True)
            if step in theory_steps:
                info = {"step": step, **theory_update(model, named_alpha, probe_loader, device, args)}
                theory_log.append(info)
                print(f"[{args.run_id}] alpha-theory @ step {step}: {info}", flush=True)
                if full is not None and step == max(theory_steps):     # --probe_reinsert
                    train_loader, restart = loader(full, True, args.seed + 1), True
                    print(f"[{args.run_id}] probe rejoins training: train={len(full)}", flush=True)
            if step % eval_every == 0 or step == total:
                d, t = evaluate(model, dev_loader, device), evaluate(model, test_loader, device)
                rec = {"step": step, "loss": float(loss.item()), "dev": d, "test": t}
                if alpha:
                    v = torch.cat([p.detach().flatten() for p in alpha])
                    v = v.exp() if args.alpha_param == "log" else v
                    rec.update(alpha_mean=float(v.mean()), alpha_std=float(v.std()),
                               alpha_min=float(v.min()), alpha_max=float(v.max()))
                history.append(rec)
                if d > best["dev"]:
                    best = {"dev": d, "test": t, "step": step}
                json.dump({"run_id": args.run_id, "step": step, "total_steps": total,
                           "elapsed_min": (time.time() - t0) / 60, "last": rec, "best": best}, open(prog_path, "w"))
                print(f"[{args.run_id}] step {step}/{total} loss={loss.item():.4f} dev={d:.4f} test={t:.4f} "
                      f"best_dev={best['dev']:.4f} (test {best['test']:.4f})"
                      + (f" alpha={rec['alpha_mean']:.3f}±{rec['alpha_std']:.3f}" if alpha else ""), flush=True)
            if step >= total or restart:
                break

    res = {"args": vars(args), "run_id": args.run_id, "test_at_best_dev": best["test"], "best_dev": best["dev"],
           "best_step": best["step"], "test_final": history[-1]["test"], "trainable": n_ad, "alpha_params": n_al,
           "runtime_min": (time.time() - t0) / 60, "history": history, "theory_log": theory_log}
    json.dump(res, open(final_path + ".tmp", "w"), indent=1)
    os.replace(final_path + ".tmp", final_path)
    if os.path.exists(prog_path):
        os.remove(prog_path)
    print(f"[{args.run_id}] DONE test@best_dev={best['test']:.4f} final={history[-1]['test']:.4f} "
          f"({res['runtime_min']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
