"""Tables for the commonsense-reasoning experiments.

  python commonsense_reasoning/summarize.py commonsense_reasoning/results/main
Metric: test accuracy (%) at the checkpoint with the best DEV accuracy, mean ± std over seeds.
If a configuration was run with several LRs, the LR is selected per configuration on mean DEV accuracy.
Writes <dir>/summary.md (also printed) and <dir>/summary.csv.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SHORT = {"boolq": "BoolQ", "piqa": "PIQA", "siqa": "SIQA", "hellaswag": "HellaS", "winogrande": "WinoG",
         "arc_e": "ARC-e", "arc_c": "ARC-c", "obqa": "OBQA", "avg": "Avg"}
ORDER = ["zero-shot", "LoRA", "LoRA+", "DoRA", "IA3", "alpha-LoRA-grad", "alpha-LoRA-grad-v2", "alpha-LoRA-theory",
         "alpha-LoRA-grad-v2+LoRA+", "alpha-LoRA-theory+LoRA+"]


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "*.json")):
        if f.endswith(".progress.json"):
            continue
        r = json.load(open(f)); a = r["args"]
        base = dict(group=a["group"], model=a["model"].split("/")[-1], label=a["label"], lr=a["lr"], seed=a["seed"],
                    n=a["n_per_task"], dev=100 * r["best_dev"], runtime=r["runtime_min"], trainable=r["trainable"],
                    alpha_params=r["alpha_params"], total=r["total_params"],
                    alpha_dev=(r.get("alpha_at_best") or {}).get("alpha_mean_abs_dev", np.nan),
                    theory_gain=np.mean([t["pred_acc_after"] - t["pred_acc_before"] for t in r.get("theory_log", [])
                                         if not t.get("skipped")] or [np.nan]))
        for k, v in r["test_best"].items():
            base[k] = 100 * v
        base["avg_final"] = 100 * r["test_final"]["avg"]
        rows.append(base)
    return pd.DataFrame(rows)


def progress(d):
    out = []
    st = os.path.join(d, "..", "..", "logs", os.path.basename(os.path.normpath(d)), "_queue_status.json")
    if os.path.exists(st):
        s = json.load(open(st))
        out.append(f"queue: {s['done']}/{s['total']} done | running {len(s['running'])} | pending {s['pending']} | "
                   f"failed {len(s['failed'])} {s['failed'][:4]} | {s['time']}")
    for f in sorted(glob.glob(os.path.join(d, "*.progress.json"))):
        p = json.load(open(f))
        out.append(f"  {p['run_id']}: step {p['step']}/{p['total_steps']} ({p['elapsed_min']:.0f} min) "
                   f"dev={p['last']['dev']:.4f} best_dev={p['best_dev']:.4f}")
    return "\n".join(out)


def ms(x):
    return f"{np.mean(x):.2f} ± {np.std(x):.2f}" if len(x) > 1 else f"{np.mean(x):.2f}"


def tables(df):
    md = []
    tasks = [c for c in SHORT if c in df.columns]
    for (g, mdl), sub in df.groupby(["group", "model"]):
        multi_n = sub["n"].nunique() > 1
        keys = ["label", "n", "lr"]
        dev = sub.groupby(keys)["dev"].mean().reset_index()
        best = dev.loc[dev.groupby(["label", "n"])["dev"].idxmax()]
        sel = sub.merge(best[keys], on=keys)
        labels = sorted(sel["label"].unique(), key=lambda l: (ORDER.index(l) if l in ORDER else 99, l))
        if not multi_n:
            rows = []
            ref = sel[sel.label == "LoRA"]["avg"].mean() if "LoRA" in set(sel.label) else np.nan
            for l in labels:
                s = sel[sel.label == l]
                row = {"method": l, **{SHORT[t]: (f"{s[t].mean():.2f}" if t != "avg" else ms(s[t].values)) for t in tasks}}
                row["Δ vs LoRA"] = f"{s['avg'].mean() - ref:+.2f}" if not np.isnan(ref) else ""
                row["lr"] = s["lr"].iloc[0]; row["seeds"] = len(s)
                if s["alpha_dev"].notna().any():
                    row["mean|α-1|"] = f"{s['alpha_dev'].mean():.3f}"
                if s["theory_gain"].notna().any():
                    row["pred. probe gain"] = f"{100*s['theory_gain'].mean():+.2f}"
                rows.append(row)
            md.append(f"\n### {g} | {mdl} | test acc (%) @ best dev, n={sub['n'].iloc[0]}/task\n")
            md.append(pd.DataFrame(rows).fillna("").to_markdown(index=False))
            if (sub.groupby("label")["lr"].nunique() > 1).any():
                allr = sub.groupby(["label", "lr"]).agg(dev=("dev", "mean"), avg=("avg", "mean"), n=("seed", "count")).round(2)
                md.append("\nall LRs (dev / test avg):\n")
                md.append(allr.to_markdown())
        else:
            piv = sel.groupby(["label", "n"])["avg"].apply(lambda x: ms(x.values)).unstack("n")
            piv = piv.loc[[l for l in labels if l in piv.index]]
            md.append(f"\n### {g} | {mdl} | test Avg (%) vs. training examples per task\n")
            md.append(piv.to_markdown())
            if "LoRA" in set(sel.label):
                lora = sel[sel.label == "LoRA"].groupby("n")["avg"].mean()
                gain = sel.groupby(["label", "n"])["avg"].mean().unstack("n").sub(lora, axis=1).round(2)
                md.append("\ngain over LoRA (points):\n")
                md.append(gain.loc[[l for l in labels if l in gain.index]].to_markdown())
    ov = df[df.alpha_params > 0].groupby("model").agg(alpha=("alpha_params", "first"), lora=("trainable", "first"),
                                                       total=("total", "first")).reset_index()
    if len(ov):
        ov["α % of trainable"] = (100 * ov.alpha / ov.lora).round(2)
        ov["α % of total"] = (100 * ov.alpha / ov.total).round(4)
        md.append("\n### parameter overhead of alpha\n")
        md.append(ov.to_markdown(index=False))
    return "\n".join(md)


def select_lr(sub):
    keys = ["label", "n", "lr"]
    dev = sub.groupby(keys)["dev"].mean().reset_index()
    best = dev.loc[dev.groupby(["label", "n"])["dev"].idxmax()]
    return sub.merge(best[keys], on=keys)


def brief(df, name):
    """One compact line per method: test Avg, gain over LoRA (headline view for monitor.sh all)."""
    out = []
    for (g, mdl), sub in df.groupby(["group", "model"]):
        sel = select_lr(sub)
        n = sel["n"].max()                       # largest data budget if several
        sel = sel[sel.n == n]
        ref = sel[sel.label == "LoRA"]["avg"].mean() if "LoRA" in set(sel.label) else np.nan
        agg = sel.groupby("label")["avg"].agg(["mean", "std", "count"])
        agg = agg.loc[sorted(agg.index, key=lambda l: (ORDER.index(l) if l in ORDER else 99, l))]
        out.append(f"  [{name}/{g} | {mdl} | n={n}/task]")
        for l, r in agg.iterrows():
            d = f"{r['mean'] - ref:+.2f}" if not np.isnan(ref) else "   "
            out.append(f"    {l:32s} {r['mean']:6.2f} ± {0 if np.isnan(r['std']) else r['std']:4.2f}  Δ LoRA {d}  ({int(r['count'])} seeds)")
    return "\n".join(out)


def plots(df, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(d, "plots"), exist_ok=True)
    hist = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        if not f.endswith(".progress.json"):
            r = json.load(open(f)); hist[r["run_id"]] = r
    for (g, mdl), sub in df.groupby(["group", "model"]):
        sel = select_lr(sub)
        labels = sorted(sel["label"].unique(), key=lambda l: (ORDER.index(l) if l in ORDER else 99, l))
        tag = f"{g}_{mdl}"
        if sel["n"].nunique() > 1:                                     # scarcity curves
            fig, ax = plt.subplots(figsize=(6, 4))
            for l in labels:
                s = sel[sel.label == l].groupby("n")["avg"].agg(["mean", "std"]).reset_index()
                ax.errorbar(s["n"], s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=3, label=l)
            ax.set_xscale("log"); ax.set_xlabel("training examples per task"); ax.set_ylabel("test Avg (%)")
            ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout()
            fig.savefig(os.path.join(d, "plots", f"{tag}_scarcity.png"), dpi=150); plt.close(fig)
        else:                                                          # bar chart
            agg = sel.groupby("label")["avg"].agg(["mean", "std"]).loc[labels]
            fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels)), 4))
            ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"].fillna(0), capsize=3,
                   color=["#c44" if ("alpha" in l or l.startswith("aLoRA")) else "#888" for l in agg.index])
            ax.set_xticks(range(len(agg))); ax.set_xticklabels(agg.index, rotation=35, ha="right", fontsize=8)
            lo = agg["mean"].min() - 3 * max(agg["std"].fillna(0).max(), 0.5)
            ax.set_ylim(max(0, lo), agg["mean"].max() + 2); ax.set_ylabel("test Avg (%)")
            ax.grid(axis="y", alpha=.3); fig.tight_layout()
            fig.savefig(os.path.join(d, "plots", f"{tag}_bars.png"), dpi=150); plt.close(fig)
        # dev curves (mean over seeds, selected LR, largest n)
        fig, ax = plt.subplots(figsize=(6, 4))
        nmax = sel["n"].max()
        for l in labels:
            runs = [hist[rid] for rid in hist if hist[rid]["args"]["label"] == l and hist[rid]["args"]["group"] == g
                    and hist[rid]["args"]["n_per_task"] == nmax and hist[rid]["history"]
                    and hist[rid]["args"]["lr"] == sel[sel.label == l]["lr"].iloc[0]]
            if runs:
                L = min(len(r["history"]) for r in runs)
                steps = [h["step"] for h in runs[0]["history"][:L]]
                ax.plot(steps, 100 * np.mean([[h["dev"] for h in r["history"][:L]] for r in runs], 0), marker=".", label=l)
        ax.set_xlabel("step"); ax.set_ylabel("dev accuracy (%)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(d, "plots", f"{tag}_dev_curves.png"), dpi=150); plt.close(fig)
    return os.path.join(d, "plots")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--brief", action="store_true", help="only print the headline lines (method, Avg, gain over LoRA)")
    ap.add_argument("--plots", action="store_true", help="also write PNG figures to <dir>/plots/")
    a = ap.parse_args()
    df = load(a.dir)
    if a.brief:
        print(brief(df, os.path.basename(os.path.normpath(a.dir))) if len(df) else "  no finished run yet")
        return
    md = progress(a.dir) + "\n"
    if len(df):
        md += tables(df)
        df.to_csv(os.path.join(a.dir, "summary.csv"), index=False)
        if a.plots:
            md += f"\n\nplots written to {plots(df, a.dir)}"
    else:
        md += "no finished run yet"
    open(os.path.join(a.dir, "summary.md"), "w").write(md)
    if not a.quiet:
        print(md)


if __name__ == "__main__":
    main()
