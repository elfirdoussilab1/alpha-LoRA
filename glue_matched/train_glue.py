"""LoRA vs alpha-LoRA on GLUE (roberta-base) with IDENTICAL hyper-parameters.

Answers the reviewers' concern (AC bullet 2, LUNR (1), 4qhw W3, AI review): everything shared by the two methods
is the same -- adapter learning rate, schedule, epochs, batch size, LoRA rank/targets, seeds, data splits and
checkpoint rule -- for every task and every seed. The ONLY difference is the alpha vector of alpha-LoRA
(W = diag(alpha) W* + BA), trained as in the paper (Algorithm 1): own Adam optimizer, one step every T adapter
steps on a fresh batch; its hyper-parameters (T, lr_alpha) are also fixed for all tasks and seeds.

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
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
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
    ap.add_argument("--method", default="lora", choices=["lora", "alpha-lora"])
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
    # protocol / io
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--dev_max", type=int, default=2000)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--num_evals", type=int, default=20)
    ap.add_argument("--out_dir", default="glue_matched/results")
    ap.add_argument("--run_id", default=None)
    a = ap.parse_args()
    a.run_id = a.run_id or f"{a.task}_{a.method}_lr{a.lr}_s{a.seed}"
    return a


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_data(args, tok):
    keys, test_split, num_labels, max_len = TASKS[args.task]
    raw = load_dataset("nyu-mll/glue", args.task)
    n_dev = int(min(args.dev_max, max(200, args.dev_frac * len(raw["train"]))))
    sp = raw["train"].train_test_split(test_size=n_dev, seed=args.data_seed)

    def prep(ds):
        ds = ds.map(lambda ex: tok(ex[keys[0]], ex[keys[1]] if keys[1] else None, truncation=True, max_length=max_len),
                    batched=True)
        keep = {"input_ids", "attention_mask", "label"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep]).rename_column("label", "labels")
        ds.set_format("torch")
        return ds
    return prep(sp["train"]), prep(sp["test"]), prep(raw[test_split]), num_labels


def attach_alpha(model):
    """alpha-LoRA: one scale per output neuron of every frozen matrix adapted by LoRA (applied in fp32)."""
    params = []
    for _, mod in model.named_modules():
        if isinstance(mod, LoraLayer) and not hasattr(mod, "alpha_scale"):
            base = mod.get_base_layer()
            p = nn.Parameter(torch.ones(base.out_features, device=base.weight.device))
            mod.register_parameter("alpha_scale", p)
            base.register_forward_hook(lambda m, i, o, p=p: o.float() * p)
            params.append(p)
    return params


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


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, f"{args.run_id}.json")
    prog_path = os.path.join(args.out_dir, f"{args.run_id}.progress.json")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    train, dev, test, num_labels = build_data(args, tok)
    coll = DataCollatorWithPadding(tok)
    loader = lambda ds, shuf, seed=0, bs=args.batch_size: DataLoader(
        ds, batch_size=bs, shuffle=shuf, collate_fn=coll, drop_last=shuf, generator=torch.Generator().manual_seed(seed))
    train_loader = loader(train, True, args.seed)
    dev_loader, test_loader = loader(dev, False, bs=128), loader(test, False, bs=128)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha,
                                             lora_dropout=args.lora_dropout,
                                             target_modules=args.target_modules.split(",")))
    alpha = attach_alpha(model) if args.method == "alpha-lora" else []
    model.to(device)
    a_ids = {id(p) for p in alpha}
    adapter = [p for p in model.parameters() if p.requires_grad and id(p) not in a_ids]   # LoRA + classifier head

    total = int(math.ceil(args.epochs * len(train_loader)))
    opt = torch.optim.AdamW(adapter, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    opt_alpha = torch.optim.Adam(alpha, lr=args.lr_alpha, betas=(0.9, 0.99)) if alpha else None

    def fresh():
        g = torch.Generator().manual_seed(args.seed + 7919)
        while True:
            for b in DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=coll, drop_last=True, generator=g):
                yield b
    fresh_it = fresh()

    def loss_on(b):
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            return model(**b).loss

    n_ad, n_al = sum(p.numel() for p in adapter), sum(p.numel() for p in alpha)
    print(f"[{args.run_id}] train={len(train)} dev={len(dev)} test={len(test)} steps={total} "
          f"trainable={n_ad} alpha={n_al} ({100 * n_al / n_ad:.2f}% of trainable)", flush=True)
    eval_every = max(1, total // args.num_evals)
    best, history, step, t0 = {"dev": -1.0}, [], 0, time.time()
    model.train()
    while step < total:
        for b in train_loader:
            loss = loss_on(b)
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            for p in alpha:
                p.grad = None
            step += 1
            if opt_alpha is not None and step % args.T == 0:          # Algorithm 1 of the paper
                grads = torch.autograd.grad(loss_on(next(fresh_it)), alpha)
                for p, g in zip(alpha, grads):
                    p.grad = g
                opt_alpha.step(); opt_alpha.zero_grad(set_to_none=True)
            if step % eval_every == 0 or step == total:
                d, t = evaluate(model, dev_loader, device), evaluate(model, test_loader, device)
                rec = {"step": step, "loss": float(loss.item()), "dev": d, "test": t}
                if alpha:
                    v = torch.cat([p.detach().flatten() for p in alpha])
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
            if step >= total:
                break

    res = {"args": vars(args), "run_id": args.run_id, "test_at_best_dev": best["test"], "best_dev": best["dev"],
           "best_step": best["step"], "test_final": history[-1]["test"], "trainable": n_ad, "alpha_params": n_al,
           "runtime_min": (time.time() - t0) / 60, "history": history}
    json.dump(res, open(final_path + ".tmp", "w"), indent=1)
    os.replace(final_path + ".tmp", final_path)
    if os.path.exists(prog_path):
        os.remove(prog_path)
    print(f"[{args.run_id}] DONE test@best_dev={best['test']:.4f} final={history[-1]['test']:.4f} "
          f"({res['runtime_min']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
