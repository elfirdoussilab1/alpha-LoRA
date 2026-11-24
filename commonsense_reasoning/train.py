"""Fine-tune a decoder LLM on the commonsense-reasoning suite with LoRA / LoRA+ / DoRA / IA3 / alpha-LoRA.

alpha-LoRA:  W_new = diag(alpha) W* + (LoRA update), alpha = one scale per output neuron of each frozen matrix
(implemented as a forward hook on the frozen base layer, so it composes with LoRA and LoRA+).

  --alpha grad    alpha-LoRA-grad: alpha learned by gradient descent with its own optimizer.
                  Paper version (default flags): every T steps, one step on a fresh batch of the training pool.
                  v2 (proposed improvements, all optional flags, see run_main.sh):
                    --alpha_data heldout   alpha is fitted on a held-out PROBE split never seen by the adapter
                                           (alpha* maximises *test* accuracy in the theory; fresh batches from
                                           the training pool are not held out after the first epoch)
                    --alpha_start 0.2      alpha frozen during the first 20% of training (alpha* is defined
                                           w.r.t. a trained adapter; early alpha steps chase a random adapter)
                    --alpha_param log      alpha = exp(theta): positive, scale-invariant (multiplicative) steps
                    --alpha_accum 4        average the alpha gradient over 4 held-out batches (less noise)
                    --alpha_objective sharpe   minimise -mean(margin)/std(margin) on held-out batches, i.e.
                                           maximise the accuracy formula of Prop. 4.3 instead of the CE loss

  --alpha theory  alpha-LoRA-theory: NO gradient descent on alpha. At given fractions of training:
                  1. on the held-out probe split, compute for each example the margin
                       d(x) = logit_correct - logit_best_wrong
                     and its derivatives d_j(x) = d d(x) / d alpha_j for K groups of modules (one batched
                     forward/backward pass, per-example values captured with hooks);
                  2. the fine-tuned predictor is affine in alpha (eq. alpha-FTC), so locally
                       d(x; alpha + Delta) = d(x) + sum_j Delta_j d_j(x)   (Gaussian, Thm 4.2)
                     and the accuracy is Phi(mean/std) (Prop 4.3). Maximising mean/std over Delta has the
                     closed form of Thm 4.4 (K = 1) / Appendix D (K > 1):
                       u* ∝ Sigma^{-1} mu,  mu = E[z], Sigma = Cov[z], z = (d, d_1, ..., d_K),  Delta* = u*_{1:} / u*_0
                     (Sigma shrunk towards its diagonal, Delta clipped to a trust region);
                  3. set alpha <- alpha + Delta* (fixed), and continue LoRA training so the adapter re-adapts.

Protocol: dev split (from train) for checkpoint/LR selection only; official val/test splits for test;
identical hyper-parameters for all seeds. Writes <out_dir>/<run_id>.json at the end (resumable queue marker)
and <out_dir>/<run_id>.progress.json during training.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from data import LETTERS, TASKS, build_splits

def phi(x):  # standard normal CDF
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--tasks", default=",".join(TASKS))
    # data
    ap.add_argument("--n_per_task", type=int, default=500)
    ap.add_argument("--dev_per_task", type=int, default=100)
    ap.add_argument("--max_test", type=int, default=500)
    ap.add_argument("--probe_frac", type=float, default=0.1)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=512)
    # method
    ap.add_argument("--method", default="lora", choices=["none", "lora", "dora", "ia3"])
    ap.add_argument("--loraplus_ratio", type=float, default=0.0, help=">0: LoRA+ (lr of B = ratio * lr)")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default=",".join(ALL_LINEAR))
    ap.add_argument("--alpha", default="none", choices=["none", "grad", "theory"])
    ap.add_argument("--alpha_granularity", default="vector", choices=["vector", "module"])
    # alpha-LoRA-grad
    ap.add_argument("--alpha_data", default="fresh", choices=["fresh", "heldout"])
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--lr_alpha", type=float, default=5e-3)
    ap.add_argument("--alpha_start", type=float, default=0.0)
    ap.add_argument("--alpha_accum", type=int, default=1)
    ap.add_argument("--alpha_param", default="linear", choices=["linear", "log"])
    ap.add_argument("--alpha_objective", default="ce", choices=["ce", "sharpe"])
    # alpha-LoRA-theory
    ap.add_argument("--theory_at", default="0.3,0.6", help="fractions of training at which alpha is re-estimated")
    ap.add_argument("--theory_groups", default="global", choices=["global", "type", "layer", "module"])
    ap.add_argument("--theory_shrink", type=float, default=0.1)
    ap.add_argument("--theory_trust", type=float, default=0.25)
    # optimisation
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="debug: fp32 weights, no bf16 autocast")
    ap.add_argument("--num_evals", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    # io
    ap.add_argument("--out_dir", default="commonsense_reasoning/results/debug")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--group", default="main")
    ap.add_argument("--wandb_project", default=None)
    a = ap.parse_args()
    if a.alpha != "none" and a.method != "lora":
        raise ValueError("alpha rescaling is defined on top of LoRA (/LoRA+)")
    if a.alpha == "theory":
        a.alpha_param = "linear"
    a.label = a.label or (a.method + ("" if a.alpha == "none" else f"+alpha-{a.alpha}"))
    a.run_id = a.run_id or f"{a.model.split('/')[-1]}_{a.label}_s{a.seed}"
    return a


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------------------------- data
def letter_ids(tok):
    base = tok("Answer:", add_special_tokens=False).input_ids
    ids = []
    for L in LETTERS:
        full = tok(f"Answer: {L}", add_special_tokens=False).input_ids
        assert full[:len(base)] == base, "tokenizer merges 'Answer:' with the letter"
        ids.append(full[-1])
    assert len(set(ids)) == len(ids)
    return ids


class MCData(torch.utils.data.Dataset):
    def __init__(self, exs, tok, max_len):
        enc = tok([e["prompt"] for e in exs], truncation=True, max_length=max_len)["input_ids"]
        self.items = [(torch.tensor(x), e["label"], e["K"], e["task"]) for x, e in zip(enc, exs)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id):
    def collate(batch):
        L = max(len(b[0]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, :len(b[0])] = b[0]; mask[i, :len(b[0])] = 1
        return {"ids": ids, "mask": mask, "label": torch.tensor([b[1] for b in batch]),
                "K": torch.tensor([b[2] for b in batch]), "task": torch.tensor([b[3] for b in batch])}
    return collate


# ----------------------------------------------------------------------------------------------- model
class AlphaState:
    amp = True
    probe = False
    acc = {}


AS = AlphaState()


def make_hook(p, name, param):
    def hook(m, inp, out):
        # apply alpha in fp32: in bf16, values near 1 are spaced by 2^-7, so alpha = 1.003 would be rounded to 1
        s = p.exp() if param == "log" else p
        y = out.float() * s
        if AS.probe and y.requires_grad:
            z = out.detach()
            y.register_hook(lambda g, z=z: AS.acc.__setitem__(name, (g.float() * z.float()).flatten(1).sum(1)))
        return y
    return hook


def attach_alpha(model, granularity, param):
    from peft.tuners.lora import LoraLayer
    mods = []
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLayer) and not hasattr(mod, "alpha_scale"):
            base = mod.get_base_layer()
            if not isinstance(base, nn.Linear):
                continue
            dim = base.out_features if granularity == "vector" else 1
            p = nn.Parameter(torch.full((dim,), 0.0 if param == "log" else 1.0, device=base.weight.device))
            mod.register_parameter("alpha_scale", p)
            base.register_forward_hook(make_hook(p, name, param))
            mods.append((name, p))
    assert mods, "no LoRA layer found"
    return mods


def build_model(args, tok):
    AS.amp = not args.fp32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if AS.amp else torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    if args.grad_ckpt and args.method != "none":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    alpha_mods = []
    if args.method != "none":
        from peft import IA3Config, LoraConfig, TaskType, get_peft_model
        targets = args.target_modules.split(",")
        if args.method == "ia3":
            cfg = IA3Config(task_type=TaskType.CAUSAL_LM, target_modules=["k_proj", "v_proj", "down_proj"],
                            feedforward_modules=["down_proj"])
        else:
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout, target_modules=targets, use_dora=args.method == "dora")
        model = get_peft_model(model, cfg)
        if args.alpha != "none":
            alpha_mods = attach_alpha(model, args.alpha_granularity, args.alpha_param)
            if args.alpha == "theory":
                for _, p in alpha_mods:
                    p.requires_grad_(False)
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
    return model, alpha_mods


def last_logits(model, batch, device):
    """Logits of the next token after 'Answer:' (lm_head applied to the last real position only)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    ids, mask = batch["ids"].to(device), batch["mask"].to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=AS.amp):
        h = base.model(input_ids=ids, attention_mask=mask).last_hidden_state
        hl = h[torch.arange(h.size(0), device=device), mask.sum(1) - 1]
        return base.lm_head(hl).float()


def letter_logits(logits, batch, lid, device):
    lett = logits[:, lid]                                           # [b, 5]
    K = batch["K"].to(device)
    valid = torch.arange(len(lid), device=device)[None, :] < K[:, None]
    return lett.masked_fill(~valid, -1e4)


def margins(lett, y):
    ar = torch.arange(len(y), device=lett.device)
    corr = lett[ar, y]
    other = lett.clone(); other[ar, y] = -1e9
    return corr - other.max(1).values


@torch.no_grad()
def evaluate(model, loader, lid, device, n_tasks):
    model.eval()
    correct, count = np.zeros(n_tasks), np.zeros(n_tasks)
    for b in loader:
        lett = letter_logits(last_logits(model, b, device), b, lid, device)
        ok = (lett.argmax(1).cpu() == b["label"]).numpy()
        np.add.at(correct, b["task"].numpy(), ok); np.add.at(count, b["task"].numpy(), 1)
    model.train()
    acc = correct / np.maximum(count, 1)
    return acc, float(acc.mean())


# ----------------------------------------------------------------------------------------------- alpha-LoRA-theory
def group_index(names, how):
    keys = []
    for n in names:
        m = re.search(r"layers\.(\d+)\.", n)
        layer = int(m.group(1)) if m else -1
        typ = n.split(".")[-1]
        keys.append({"global": 0, "type": typ, "layer": layer, "module": n}[how])
    uniq = sorted(set(keys), key=str)
    return [uniq.index(k) for k in keys], len(uniq)


def theory_update(model, alpha_mods, probe_loader, lid, device, args):
    names = [n for n, _ in alpha_mods]
    params = [p for _, p in alpha_mods]
    gidx, K = group_index(names, args.theory_groups)
    Gm = torch.zeros(len(names), K, device=device)
    Gm[torch.arange(len(names)), torch.tensor(gidx)] = 1.0
    saved = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    # stay in train mode: HF only applies gradient checkpointing when model.training is True, and without it the
    # backward pass over the probe batch keeps every layer's activations (OOM on 96 GB). Only dropout is switched off.
    drop = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    model.train()
    for m in drop:
        m.eval()
    AS.probe = True
    Z = []
    for b in probe_loader:
        AS.acc = {}
        lett = letter_logits(last_logits(model, b, device), b, lid, device)
        d = margins(lett, b["label"].to(device))
        torch.autograd.grad(d.sum(), params)
        D = torch.stack([AS.acc[n] for n in names], 1)            # [b, modules]  d d(x) / d alpha_module
        Z.append(torch.cat([d.detach()[:, None], D @ Gm], 1).double().cpu())
    AS.probe = False
    for p in model.parameters():
        p.requires_grad_(saved[id(p)])
    model.train()
    Z = torch.cat(Z).numpy()
    mu, S = Z.mean(0), np.cov(Z, rowvar=False)
    S = np.atleast_2d(S)
    Ss = S + args.theory_shrink * np.diag(np.diag(S)) + 1e-12 * np.eye(len(mu))
    u = np.linalg.solve(Ss, mu)
    info = {"n_probe": len(Z), "groups": K, "pred_acc_before": float(phi(mu[0] / np.sqrt(S[0, 0])))}
    if u[0] <= 0:
        info.update(skipped=True)
        return info
    delta = np.clip(u[1:] / u[0], -args.theory_trust, args.theory_trust)
    v = np.concatenate([[1.0], delta])
    info.update(skipped=False, pred_acc_after=float(phi(mu @ v / np.sqrt(v @ S @ v))),
                delta_mean=float(delta.mean()), delta_min=float(delta.min()), delta_max=float(delta.max()),
                frac_clipped=float(np.mean(np.abs(u[1:] / u[0]) > args.theory_trust)))
    with torch.no_grad():
        for p, g in zip(params, gidx):
            p.add_(float(delta[g]))
    return info


# ----------------------------------------------------------------------------------------------- main
def alpha_values(alpha_mods, param):
    if not alpha_mods:
        return None
    v = torch.cat([p.detach().float().flatten() for _, p in alpha_mods])
    return (v.exp() if param == "log" else v).cpu().numpy()


def alpha_stats(alpha_mods, param):
    v = alpha_values(alpha_mods, param)
    if v is None:
        return {}
    return {"alpha_mean": float(v.mean()), "alpha_std": float(v.std()), "alpha_min": float(v.min()),
            "alpha_max": float(v.max()), "alpha_mean_abs_dev": float(np.abs(v - 1).mean()), "alpha_count": int(v.size)}


# ----------------------------------------------------------------------------------------------- run cache
# Identical configurations appear in several experiments (e.g. LoRA at 500 ex/task in main AND scarcity, the
# reference rows of the ablation). With one GPU we never train the same configuration twice: results are cached
# by a hash of every argument that changes the computation, and re-labelled for the requesting experiment.
NON_SCIENTIFIC = {"out_dir", "run_id", "label", "group", "wandb_project"}


def cache_path(args):
    key = json.dumps({k: v for k, v in sorted(vars(args).items()) if k not in NON_SCIENTIFIC}, sort_keys=True)
    root = os.path.join(os.path.dirname(os.path.normpath(args.out_dir)), "_cache")
    return os.path.join(root, hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")


def try_reuse(args, final_path):
    cp = cache_path(args)
    if not os.path.exists(cp):
        return False
    res = json.load(open(cp))
    src = res["run_id"]
    res["args"].update({k: getattr(args, k) for k in NON_SCIENTIFIC})
    res.update(run_id=args.run_id, reused_from=src)
    json.dump(res, open(final_path, "w"), indent=1)
    print(f"[{args.run_id}] identical configuration already trained ({src}) -> result reused", flush=True)
    return True


def save_result(args, res, final_path):
    tmp = final_path + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, final_path)
    cp = cache_path(args)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    json.dump(res, open(cp, "w"))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, f"{args.run_id}.json")
    if try_reuse(args, final_path):
        return
    prog_path = os.path.join(args.out_dir, f"{args.run_id}.progress.json")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = args.tasks.split(",")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"          # long passages are cut from the left: options + 'Answer:' are kept
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lid = letter_ids(tok)
    sp = build_splits(tasks, args.n_per_task, args.dev_per_task, args.max_test, args.probe_frac,
                      args.data_seed, subset_seed=args.seed)
    uses_probe = args.alpha == "theory" or (args.alpha == "grad" and args.alpha_data == "heldout")
    train_exs = sp["train"] if uses_probe else sp["train"] + sp["probe"]   # same labelled data for every method
    coll = make_collate(tok.pad_token_id)
    train_ds = MCData(train_exs, tok, args.max_len)
    mk = lambda ds, bs, shuf, seed=0: DataLoader(ds, batch_size=bs, shuffle=shuf, collate_fn=coll, drop_last=shuf,
                                                 generator=torch.Generator().manual_seed(seed))
    sort = lambda exs: sorted(exs, key=lambda e: len(e["prompt"]))
    train_loader = mk(train_ds, args.batch_size, True, args.seed)
    dev_loader = mk(MCData(sort(sp["dev"]), tok, args.max_len), args.eval_bs, False)
    test_loader = mk(MCData(sort(sp["test"]), tok, args.max_len), args.eval_bs, False)
    probe_ds = MCData(sp["probe"], tok, args.max_len) if sp["probe"] else None

    model, alpha_mods = build_model(args, tok)
    model.to(device)
    n_tasks = len(tasks)
    t0 = time.time()

    if args.method == "none":                                   # zero-shot reference
        dev_acc, dev_avg = evaluate(model, dev_loader, lid, device, n_tasks)
        acc, avg = evaluate(model, test_loader, lid, device, n_tasks)
        res = {"args": vars(args), "run_id": args.run_id, "tasks": tasks, "best_dev": dev_avg,
               "test_best": dict(zip(tasks, acc.tolist()), avg=avg), "test_final": dict(zip(tasks, acc.tolist()), avg=avg),
               "trainable": 0, "alpha_params": 0, "total_params": sum(p.numel() for p in model.parameters()),
               "runtime_min": (time.time() - t0) / 60, "history": []}
        save_result(args, res, final_path)
        print(f"[{args.run_id}] zero-shot test avg={avg:.4f}", flush=True)
        return

    alpha_params = [p for _, p in alpha_mods]
    a_ids = {id(p) for p in alpha_params}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and id(p) not in a_ids]
    if args.loraplus_ratio > 0:
        groups = [{"params": [p for n, p in named if "lora_B" not in n], "lr": args.lr},
                  {"params": [p for n, p in named if "lora_B" in n], "lr": args.lr * args.loraplus_ratio}]
    else:
        groups = [{"params": [p for _, p in named], "lr": args.lr}]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total = int(math.ceil(args.epochs * steps_per_epoch))
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    opt_alpha = torch.optim.Adam(alpha_params, lr=args.lr_alpha, betas=(0.9, 0.99)) if args.alpha == "grad" else None
    eval_every = max(1, total // args.num_evals)
    theory_steps = sorted({max(1, int(float(f) * total)) for f in args.theory_at.split(",")}) if args.alpha == "theory" else []
    n_train_par = sum(p.numel() for _, p in named)
    n_alpha = sum(p.numel() for p in alpha_params)

    def batches(ds, seed):
        g = torch.Generator().manual_seed(seed)
        while True:
            for b in DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll, drop_last=True, generator=g):
                yield b
    alpha_it = None
    if args.alpha == "grad":
        alpha_it = batches(probe_ds if args.alpha_data == "heldout" else train_ds, args.seed + 7919)
    probe_loader = mk(probe_ds, args.eval_bs // 2 or 1, False) if probe_ds is not None else None
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_id, config=vars(args))

    print(f"[{args.run_id}] train={len(train_ds)} probe={len(probe_ds) if probe_ds else 0} dev={len(sp['dev'])} "
          f"test={len(sp['test'])} steps={total} trainable={n_train_par} alpha={n_alpha} "
          f"({100*n_alpha/max(n_train_par,1):.2f}% of trainable)", flush=True)

    keep = [(n, p) for n, p in model.named_parameters() if p.requires_grad or id(p) in a_ids]
    best = {"dev": -1.0, "step": 0, "state": None}
    history, theory_log = [], []
    step = 0
    model.train()
    while step < total:
        for b in train_loader:
            lett_all = last_logits(model, b, device)
            loss = F.cross_entropy(lett_all, torch.tensor(lid, device=device)[b["label"].to(device)])
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            for p in alpha_params:
                p.grad = None
            step += 1

            if opt_alpha is not None and step % args.T == 0 and step >= args.alpha_start * total:
                gsum = [torch.zeros_like(p) for p in alpha_params]
                for _ in range(args.alpha_accum):
                    ab = next(alpha_it)
                    lg = last_logits(model, ab, device)
                    if args.alpha_objective == "ce":
                        aloss = F.cross_entropy(lg, torch.tensor(lid, device=device)[ab["label"].to(device)])
                    else:
                        d = margins(letter_logits(lg, ab, lid, device), ab["label"].to(device))
                        aloss = -d.mean() / (d.std() + 1e-3)
                    for gs, g in zip(gsum, torch.autograd.grad(aloss, alpha_params)):
                        gs += g / args.alpha_accum
                for p, g in zip(alpha_params, gsum):
                    p.grad = g
                opt_alpha.step(); opt_alpha.zero_grad(set_to_none=True)

            if step in theory_steps:
                info = theory_update(model, alpha_mods, probe_loader, lid, device, args)
                info["step"] = step
                theory_log.append(info)
                print(f"[{args.run_id}] alpha-theory @ step {step}: {info}", flush=True)

            if step % eval_every == 0 or step == total:
                _, dev_avg = evaluate(model, dev_loader, lid, device, n_tasks)
                rec = {"step": step, "loss": float(loss.item()), "dev": dev_avg, **alpha_stats(alpha_mods, args.alpha_param)}
                history.append(rec)
                if dev_avg > best["dev"]:
                    best = {"dev": dev_avg, "step": step, "state": {n: p.detach().cpu().clone() for n, p in keep}}
                json.dump({"run_id": args.run_id, "step": step, "total_steps": total, "elapsed_min": (time.time() - t0) / 60,
                           "last": rec, "best_dev": best["dev"]}, open(prog_path, "w"))
                print(f"[{args.run_id}] step {step}/{total} loss={loss.item():.4f} dev={dev_avg:.4f} "
                      f"best={best['dev']:.4f}@{best['step']} "
                      + (f"alpha={rec['alpha_mean']:.3f}±{rec['alpha_std']:.3f}" if alpha_mods else ""), flush=True)
                if args.wandb_project:
                    wandb.log({k: v for k, v in rec.items() if isinstance(v, (int, float))}, step=step)
            if step >= total:
                break

    acc_f, avg_f = evaluate(model, test_loader, lid, device, n_tasks)
    a_final = alpha_stats(alpha_mods, args.alpha_param)
    with torch.no_grad():
        for n, p in keep:
            p.copy_(best["state"][n].to(p.device))
    acc_b, avg_b = evaluate(model, test_loader, lid, device, n_tasks)
    res = {"args": vars(args), "run_id": args.run_id, "tasks": tasks, "best_dev": best["dev"], "best_step": best["step"],
           "test_best": dict(zip(tasks, acc_b.tolist()), avg=avg_b), "test_final": dict(zip(tasks, acc_f.tolist()), avg=avg_f),
           "trainable": n_train_par, "alpha_params": n_alpha, "total_params": sum(p.numel() for p in model.parameters()),
           "runtime_min": (time.time() - t0) / 60,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0,
           "alpha_final": a_final, "alpha_at_best": alpha_stats(alpha_mods, args.alpha_param),
           "theory_log": theory_log, "history": history}
    save_result(args, res, final_path)
    if os.path.exists(prog_path):
        os.remove(prog_path)
    print(f"[{args.run_id}] DONE test avg @best dev={avg_b:.4f} final={avg_f:.4f} ({res['runtime_min']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
