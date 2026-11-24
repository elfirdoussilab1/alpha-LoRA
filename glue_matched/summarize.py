"""Tables for glue_matched.
  python glue_matched/summarize.py [results_dir]     (default glue_matched/results; writes <dir>/summary.md and .csv)
Metric: GLUE validation accuracy (%) at the best-dev checkpoint, mean ± std over seeds."""
import glob, json, os, sys
import numpy as np
import pandas as pd

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER = ["rte", "mrpc", "sst2", "qnli", "mnli", "qqp"]
METHODS = ["LoRA", "DoRA", "IA3", "LoRA+", "alpha-LoRA-grad", "alpha-LoRA-grad-v2", "alpha-LoRA-theory",
           "alpha-LoRA-grad+LoRA+", "alpha-LoRA-grad-v2+LoRA+", "alpha-LoRA-theory+LoRA+"]
LEGACY = {"lora": "LoRA", "alpha-lora": "alpha-LoRA-grad", "alpha-lora-theory": "alpha-LoRA-theory"}


def ms(x):
    return f"{np.mean(x):.2f} ± {np.std(x):.2f}" if len(x) > 1 else f"{np.mean(x):.2f}"


def ordered(labels):
    return sorted(labels, key=lambda l: (METHODS.index(l) if l in METHODS else 99, l))


def main():
    D = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    out = []
    st = os.path.join(os.path.dirname(D), os.path.basename(D).replace("results", "logs"), "_queue_status.json")
    if os.path.exists(st):
        s = json.load(open(st))
        out.append(f"queue: {s['done']}/{s['total']} done | running {len(s['running'])} | pending {s['pending']} | "
                   f"failed {len(s['failed'])} {s['failed'][:4]} | {s['time']}")
    for f in sorted(glob.glob(os.path.join(D, "*.progress.json"))):
        p = json.load(open(f))
        out.append(f"  {p['run_id']}: step {p['step']}/{p['total_steps']} dev={p['last']['dev']:.4f} best_dev={p['best']['dev']:.4f}")
    rows = []
    for f in glob.glob(os.path.join(D, "*.json")):
        if f.endswith(".progress.json"):
            continue
        r = json.load(open(f)); a = r["args"]
        rows.append(dict(task=a["task"], method=a.get("label") or LEGACY.get(a["method"], a["method"]), lr=a["lr"],
                         seed=a["seed"], dev=100 * r["best_dev"], test=100 * r["test_at_best_dev"],
                         final=100 * r["test_final"],
                         alpha_range=(f"[{r['history'][-1]['alpha_min']:.2f}, {r['history'][-1]['alpha_max']:.2f}]"
                                      if "alpha_min" in r["history"][-1] else "")))
    if not rows:
        out.append("no finished run yet")
    else:
        df = pd.DataFrame(rows)
        tasks = [t for t in ORDER if t in set(df.task)]

        # 1) main table: every method at its own best LR, chosen per task on DEV (mean over seeds)
        dv = df.groupby(["task", "method", "lr"])["dev"].mean().reset_index()
        best = dv.loc[dv.groupby(["task", "method"])["dev"].idxmax()][["task", "method", "lr"]]
        sel = df.merge(best, on=["task", "method", "lr"])
        mean = sel.groupby(["method", "task"])["test"].mean().unstack("task")[tasks]
        t1 = sel.groupby(["method", "task"])["test"].apply(lambda x: ms(x.values)).unstack("task")[tasks]
        t1["avg"] = mean.mean(axis=1).round(2)
        t1["avg (final ckpt)"] = sel.groupby(["method", "task"])["final"].mean().unstack("task")[tasks].mean(axis=1).round(2)
        if "LoRA" in mean.index:
            t1["Δ avg vs LoRA"] = (t1["avg"] - t1.loc["LoRA", "avg"]).map("{:+.2f}".format)
            t1["tasks > LoRA"] = mean.gt(mean.loc["LoRA"], axis=1).sum(axis=1).astype(str) + f"/{len(tasks)}"
            paired = sel.pivot_table(index=["task", "seed"], columns="method", values="test")
            t1["seeds > LoRA"] = [f"{int((paired[m] > paired['LoRA']).sum())}/{len(paired[[m, 'LoRA']].dropna())}"
                                  for m in t1.index]
        rank = mean.rank(axis=0, ascending=False, method="min")         # 1 = best method on that task
        t1["avg rank"] = rank.mean(axis=1).round(2)
        t1["best on"] = (rank == 1).sum(axis=1).astype(str) + f"/{len(tasks)}"
        out.append("\n### 1) Each method at its best LR (chosen on DEV): test acc (%) @ best dev, mean ± std over seeds\n")
        out.append(t1.loc[ordered(t1.index)].to_markdown())
        out.append("\nLR selected on dev:\n")
        out.append(best.pivot(index="method", columns="task", values="lr").loc[ordered(t1.index)][tasks].to_markdown())

        # 2) matched LR: methods that share an LR, same seeds / data
        if df.lr.nunique() > 1 or df.method.nunique() > 1:
            out.append("\n### 2) Matched hyper-parameters: test acc (%) @ best dev, at each LR\n")
            t2 = df.groupby(["lr", "method", "task"])["test"].apply(lambda x: ms(x.values)).unstack("task")[tasks]
            t2["avg"] = df.groupby(["lr", "method", "task"])["test"].mean().unstack("task")[tasks].mean(axis=1).round(2)
            out.append(t2.to_markdown())
        # paired per-seed difference of each alpha variant vs its base method (LoRA, or LoRA+ for the +LoRA+ rows)
        pv = df.pivot_table(index=["task", "lr", "seed"], columns="method", values="test")
        for m in ordered(c for c in pv.columns if c.startswith("alpha")):
            ref = "LoRA+" if m.endswith("+LoRA+") else "LoRA"
            if ref not in pv.columns:
                continue
            diff = (pv[m] - pv[ref]).dropna().rename("diff").reset_index()
            if diff.empty:
                continue
            g = diff.groupby(["lr", "task"])["diff"].agg(["mean", "std", lambda x: f"{(x > 0).sum()}/{len(x)}"])
            g.columns = [f"Δ mean ({m} − {ref})", "Δ std", "seeds won"]
            out.append(f"\npaired per-seed difference, {m} vs {ref} (same seed, same data, same LR):\n")
            out.append(g.round(2).to_markdown())
        al = df[df.alpha_range != ""].groupby(["method", "task", "lr"])["alpha_range"].first()
        if len(al):
            out.append("\nrange of learned alpha (one seed):\n")
            out.append(al.unstack("lr").to_markdown())
        df.to_csv(os.path.join(D, "summary.csv"), index=False)
    md = "\n".join(out)
    if os.path.isdir(D):
        open(os.path.join(D, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
