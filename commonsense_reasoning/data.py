"""Commonsense-reasoning benchmark (the 8-task suite used by LLM-Adapters / DoRA / LoRA variants):
BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA.

Every task is cast as lettered multiple choice:
    <question / context>
    A. <option 1>
    B. <option 2>
    ...
    Answer:            -> target token " A" / " B" / ...
so training is next-token CE on ONE token and evaluation is ONE forward pass per example (argmax over the
valid letter logits) -- much faster than generation-based evaluation. The letter rows of the frozen LM head
act as a pre-trained linear classifier on the last hidden state, i.e. exactly the alpha-FTC setting.
"""
import numpy as np
from datasets import load_dataset

LETTERS = "ABCDE"
TASKS = ["boolq", "piqa", "siqa", "hellaswag", "winogrande", "arc_e", "arc_c", "obqa"]

# task -> (hf path, config, revision, train split, test split)
SOURCES = {
    "boolq": ("google/boolq", None, None, "train", "validation"),
    "piqa": ("baber/piqa", None, None, "train", "validation"),
    "siqa": ("allenai/social_i_qa", None, "refs/convert/parquet", "train", "validation"),
    "hellaswag": ("Rowan/hellaswag", None, None, "train", "validation"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", None, "train", "validation"),
    "arc_e": ("allenai/ai2_arc", "ARC-Easy", None, "train", "test"),
    "arc_c": ("allenai/ai2_arc", "ARC-Challenge", None, "train", "test"),
    "obqa": ("allenai/openbookqa", "main", None, "train", "test"),
}


def _clip(s, n=1500):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " ..."


def _fmt(stem, options):
    return stem + "\n" + "\n".join(f"{LETTERS[i]}. {_clip(o, 400)}" for i, o in enumerate(options)) + "\nAnswer:"


def to_mc(task, ex):
    """Return (prompt, options, label) for one raw example."""
    if task == "boolq":
        return _fmt(f"{_clip(ex['passage'])}\nQuestion: {ex['question']}?", ["yes", "no"]), 2, 0 if ex["answer"] else 1
    if task == "piqa":
        return _fmt(f"Goal: {ex['goal']}\nWhich solution is better?", [ex["sol1"], ex["sol2"]]), 2, int(ex["label"])
    if task == "siqa":
        return (_fmt(f"{ex['context']}\nQuestion: {ex['question']}", [ex["answerA"], ex["answerB"], ex["answerC"]]),
                3, int(ex["label"]) - 1)
    if task == "hellaswag":
        endings = [e.replace(" [title]", ". ").strip() for e in ex["endings"]]
        return (_fmt(f"{ex['activity_label']}: {ex['ctx']}\nWhich ending is the most plausible?", endings),
                4, int(ex["label"]))
    if task == "winogrande":
        return (_fmt(f"{ex['sentence']}\nWhich option correctly fills the blank (_)?", [ex["option1"], ex["option2"]]),
                2, int(ex["answer"]) - 1)
    if task in ("arc_e", "arc_c", "obqa"):
        q = ex["question"] if task != "obqa" else ex["question_stem"]
        labels, texts = list(ex["choices"]["label"]), list(ex["choices"]["text"])
        return _fmt(f"Question: {q}", texts), len(texts), labels.index(ex["answerKey"])
    raise ValueError(task)


def _load(task, split):
    path, cfg, rev, tr, te = SOURCES[task]
    return load_dataset(path, cfg, revision=rev, split=tr if split == "train" else te)


def build_splits(tasks, n_per_task, dev_per_task, max_test, probe_frac, data_seed=0, subset_seed=0):
    """Returns dict of lists of examples {task, prompt, K, label} for 'train', 'probe', 'dev', 'test'.

    * dev  : fixed (data_seed) held-out slice of each task's TRAIN split, used only for checkpoint / LR selection
    * train: n_per_task examples per task (subset_seed, disjoint from dev)
    * probe: the last `probe_frac` of the training examples, held out from the adapter; used only by the
             alpha-LoRA variants that need held-out data (alpha-LoRA-theory, alpha-LoRA-grad-v2). Methods that
             do not use it train on train + probe, so every method sees the same labelled data.
    * test : official validation/test split of each task, capped at max_test examples (fixed subset)
    """
    out = {"train": [], "probe": [], "dev": [], "test": []}
    for ti, task in enumerate(tasks):
        tr = _load(task, "train")
        rng = np.random.default_rng(data_seed + 17 * ti)
        perm = rng.permutation(len(tr))
        dev_idx, pool = perm[:dev_per_task], perm[dev_per_task:]
        rng2 = np.random.default_rng(1000 + subset_seed + 17 * ti)
        tr_idx = rng2.permutation(pool)[:n_per_task]
        n_probe = int(round(probe_frac * len(tr_idx)))
        te = _load(task, "test")
        te_idx = np.random.default_rng(ti).permutation(len(te))[:max_test] if max_test else np.arange(len(te))
        for name, ds, idx in [("dev", tr, dev_idx), ("train", tr, tr_idx[n_probe:]), ("probe", tr, tr_idx[:n_probe]),
                              ("test", te, te_idx)]:
            for i in idx:
                p, K, y = to_mc(task, ds[int(i)])
                out[name].append({"task": ti, "prompt": p, "K": K, "label": y})
    return out


def prefetch(tasks=TASKS):
    for t in tasks:
        _load(t, "train"); _load(t, "test")
    print("datasets ok")


if __name__ == "__main__":
    s = build_splits(TASKS, 5, 2, 3, 0.2)
    for k, v in s.items():
        print(k, len(v))
    for ex in s["train"][::4]:
        print("-" * 60, "\n", ex["prompt"][-400:], "->", LETTERS[ex["label"]], f"(K={ex['K']})")
