"""Tables for glue_matched: python glue_matched/summarize.py  (writes glue_matched/results/summary.md)
Metric: GLUE validation accuracy (%) at the best-dev checkpoint, mean ± std over seeds."""
import glob, json, os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER = ["rte", "mrpc", "sst2", "qnli", "mnli", "qqp"]


def ms(x):
    return f"{np.mean(x):.2f} ± {np.std(x):.2f}" if len(x) > 1 else f"{np.mean(x):.2f}"


def main():
    out = []
    st = os.path.join(D, "..", "logs", "_queue_status.json")
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
        rows.append(dict(task=a["task"], method=a["method"], lr=a["lr"], seed=a["seed"], dev=100 * r["best_dev"],
                         test=100 * r["test_at_best_dev"], final=100 * r["test_final"],
                         alpha_range=(f"[{r['history'][-1]['alpha_min']:.2f}, {r['history'][-1]['alpha_max']:.2f}]"
                                      if "alpha_min" in r["history"][-1] else "")))
    if not rows:
        out.append("no finished run yet")
    else:
        df = pd.DataFrame(rows)
        tasks = [t for t in ORDER if t in set(df.task)]
        # 1) matched LR
        out.append("\n### 1) Matched hyper-parameters: test acc (%) @ best dev, same LR for both methods\n")
        t1 = df.groupby(["lr", "method", "task"])["test"].apply(lambda x: ms(x.values)).unstack("task")[tasks]
        t1["avg"] = df.groupby(["lr", "method"])["test"].mean().round(2)
        out.append(t1.to_markdown())
        # paired difference per seed
        pv = df.pivot_table(index=["task", "lr", "seed"], columns="method", values="test").dropna()
        if {"lora", "alpha-lora"} <= set(pv.columns):
            pv["diff"] = pv["alpha-lora"] - pv["lora"]
            g = pv.groupby(["lr", "task"])["diff"].agg(["mean", "std", lambda x: f"{(x > 0).sum()}/{len(x)}"])
            g.columns = ["Δ mean (alpha-LoRA − LoRA)", "Δ std", "seeds won"]
            out.append("\npaired per-seed difference (same seed, same data, same LR):\n")
            out.append(g.round(2).to_markdown())
        # 2) independently tuned
        if df.lr.nunique() > 1:
            dv = df.groupby(["task", "method", "lr"])["dev"].mean().reset_index()
            best = dv.loc[dv.groupby(["task", "method"])["dev"].idxmax()][["task", "method", "lr"]]
            sel = df.merge(best, on=["task", "method", "lr"])
            out.append("\n### 2) Independently tuned: each method at its best LR chosen on DEV\n")
            t2 = sel.groupby(["method", "task"])["test"].apply(lambda x: ms(x.values)).unstack("task")[tasks]
            t2["avg"] = sel.groupby("method")["test"].mean().round(2)
            out.append(t2.to_markdown())
            out.append("\nLR selected on dev:\n")
            out.append(best.pivot(index="method", columns="task", values="lr")[tasks].to_markdown())
        al = df[df.method == "alpha-lora"].groupby(["task", "lr"])["alpha_range"].first()
        if len(al):
            out.append("\nrange of learned alpha (one seed):\n")
            out.append(al.unstack("lr").to_markdown())
        df.to_csv(os.path.join(D, "summary.csv"), index=False)
    md = "\n".join(out)
    open(os.path.join(D, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
