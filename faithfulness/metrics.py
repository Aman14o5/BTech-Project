"""Faithfulness metrics with 95% bootstrap confidence intervals.

Per example x with original prediction y and variant v:
  flip(v) = 1[pred(v) != y]
  drop(v) = P(y | original) - P(y | v)      (confidence lost in the original label)

Contrasts are PAIRED: computed only on examples that have both variants, as the
mean per-example difference (cited minus control), with a bootstrap CI.
  necessity   (removal)  : flip(remove_cited) - flip(control)    > 0  -> faithful
  sufficiency (isolation): keep(isolate_cited) - keep(control)   > 0  -> faithful
Results are grouped by the model's PREDICTED label: that is the decision the
explanation is supposed to justify. Note that removing evidence is not expected
to flip a *neutral* prediction (neutral is where a model lands when evidence is
missing), so necessity is mainly interpretable for entailment/contradiction.

Label-explanation consistency: does the stance of the explanation (stance.py)
agree with the predicted label?
"""
import random

LABELS = ["entailment", "neutral", "contradiction"]
GROUPS = ["all"] + LABELS

# (name, cited variant, control variant, kind)
CONTRASTS = [
    ("premise removal vs random", "premise:remove_cited", "premise:remove_random", "necessity"),
    ("premise removal vs overlap", "premise:remove_cited", "premise:remove_overlap", "necessity"),
    ("premise isolation vs random", "premise:isolate_cited", "premise:isolate_random", "sufficiency"),
    ("hypothesis removal vs random", "hypothesis:remove_cited", "hypothesis:remove_random", "necessity"),
    ("hypothesis removal vs overlap", "hypothesis:remove_cited", "hypothesis:remove_overlap", "necessity"),
    ("gold premise removal vs random", "premise:remove_gold", "premise:remove_gold_random", "necessity"),
    ("gold hypothesis removal vs random", "hypothesis:remove_gold", "hypothesis:remove_gold_random", "necessity"),
]


# ---------------------------------------------------------------- statistics
def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    """(mean, lo, hi) of `values` with a percentile bootstrap CI."""
    values = [float(v) for v in values]
    if not values:
        return None, None, None
    m = _mean(values)
    if len(values) < 2:
        return m, m, m
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        arr = np.asarray(values)
        means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
        lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
        return m, float(lo), float(hi)
    except ImportError:
        rng = random.Random(seed)
        n_boot = min(n_boot, 200)
        means = sorted(_mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(n_boot))
        return m, means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]


def _ci_dict(values, **kw):
    m, lo, hi = bootstrap_ci(values, **kw)
    return {"mean": m, "lo": lo, "hi": hi, "n": len(values)}


def accuracy_macro_f1(gold, pred):
    acc = sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1)
    f1s = []
    for l in LABELS:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return acc, sum(f1s) / len(f1s)


def confusion(rows, cols, row_labels=LABELS, col_labels=LABELS):
    m = {r: {c: 0 for c in col_labels} for r in row_labels}
    for r, c in zip(rows, cols):
        if r in m and c in m[r]:
            m[r][c] += 1
    return m


# ---------------------------------------------------------------- per-example quantities
def _in_group(r, g):
    return g == "all" or r["original"]["label"] == g


def flip_and_drop(r, key):
    y = r["original"]["label"]
    v = r["variants"].get(key)
    if v is None:
        return None
    return float(v["label"] != y), r["original"]["probs"][y] - v["probs"][y]


# ---------------------------------------------------------------- summaries
def variant_stats(records, keys, n_boot):
    out = {}
    for key in keys:
        out[key] = {}
        for g in GROUPS:
            fd = [flip_and_drop(r, key) for r in records if _in_group(r, g)]
            fd = [x for x in fd if x is not None]
            out[key][g] = {"flip_rate": _ci_dict([f for f, _ in fd], n_boot=n_boot),
                           "prob_drop": _ci_dict([d for _, d in fd], n_boot=n_boot)}
    return out


def contrast_stats(records, n_boot):
    out = {}
    for name, a, b, kind in CONTRASTS:
        out[name] = {"cited": a, "control": b, "kind": kind}
        for g in GROUPS:
            flips, drops = [], []
            for r in records:
                if not _in_group(r, g):
                    continue
                fa, fb = flip_and_drop(r, a), flip_and_drop(r, b)
                if fa is None or fb is None:
                    continue
                # necessity: cited flips more; sufficiency: cited KEEPS more (= control flips more)
                sign = 1 if kind == "necessity" else -1
                flips.append(sign * (fa[0] - fb[0]))
                drops.append(sign * (fa[1] - fb[1]))
            out[name][g] = {"flip_diff": _ci_dict(flips, n_boot=n_boot),
                            "drop_diff": _ci_dict(drops, n_boot=n_boot)}
    return out


def transitions(records, keys):
    out = {}
    for key in keys:
        rows, cols = [], []
        for r in records:
            if key in r["variants"]:
                rows.append(r["original"]["label"])
                cols.append(r["variants"][key]["label"])
        out[key] = confusion(rows, cols)
    return out


def summarise(records, n_boot=1000):
    gold = [r["label"] for r in records]
    pred = [r["original"]["label"] for r in records]
    acc, f1 = accuracy_macro_f1(gold, pred)
    keys = sorted({k for r in records for k in r["variants"]})

    consistency = {g: _ci_dict([float(r["stance"] == r["original"]["label"]) for r in records if _in_group(r, g)],
                               n_boot=n_boot) for g in GROUPS}
    human = [r for r in records if r.get("human_stance")]

    alignment = {}
    for side in ("premise", "hypothesis"):
        alignment[side] = {}
        for g in GROUPS:
            prf = [r["cited_vs_gold"][side] for r in records if _in_group(r, g) and r["cited_vs_gold"].get(side)]
            alignment[side][g] = {m: _ci_dict([x[m] for x in prf], n_boot=n_boot) for m in ("precision", "recall", "f1")}

    return {
        "n_examples": len(records),
        "baseline": {"accuracy": acc, "macro_f1": f1, "confusion_gold_x_pred": confusion(gold, pred),
                     "pred_label_counts": {l: pred.count(l) for l in LABELS}},
        "cites_input": {side: _mean([bool(r["cited"][side]) for r in records]) for side in ("premise", "hypothesis")},
        "consistency": {
            "model": consistency,
            "model_confusion_pred_x_stance": confusion(pred, [r["stance"] for r in records],
                                                       col_labels=LABELS + ["none"]),
            "human_reference_accuracy": _mean([float(r["human_stance"] == r["label"]) for r in human]),
            "human_confusion_gold_x_stance": confusion([r["label"] for r in human], [r["human_stance"] for r in human],
                                                       col_labels=LABELS + ["none"]),
        },
        "alignment_cited_vs_gold": alignment,
        "variants": variant_stats(records, keys, n_boot),
        "contrasts": contrast_stats(records, n_boot),
        "transitions": transitions(records, keys),
    }
