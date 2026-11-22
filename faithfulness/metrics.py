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
    ("premise contradiction vs random", "premise:contradict_cited", "premise:contradict_random", "necessity"),
    ("hypothesis contradiction vs random", "hypothesis:contradict_cited", "hypothesis:contradict_random", "necessity"),
]

# Counterfactual correctness (plan, Table 2): does the prediction after an edit match the
# behaviour expected of a faithful model (plan, Table 1)?
#   removal      strict : entailment/contradiction prediction moves to neutral
#                lenient: it moves away from the original label (to anything)
#   isolation    the prediction is kept
#   contradiction the prediction equals the label an independent NLI oracle gives the edited pair
#                (only edits where the oracle is confident, p >= min_conf). Reported overall,
#                on edits that SHOULD change the label (oracle label != original prediction:
#                "follows") and on edits that should not ("stays").
REMOVAL_KEYS = ["premise:remove_cited", "premise:remove_random", "premise:remove_overlap", "premise:remove_gold",
                "hypothesis:remove_cited", "hypothesis:remove_random", "hypothesis:remove_overlap",
                "hypothesis:remove_gold"]
ISOLATION_KEYS = ["premise:isolate_cited", "premise:isolate_random", "premise:isolate_gold"]
CONTRADICTION_KEYS = ["premise:contradict_cited", "premise:contradict_random", "premise:contradict_gold",
                      "hypothesis:contradict_cited", "hypothesis:contradict_random", "hypothesis:contradict_gold"]
# cited vs control, compared on the edits that should change the label (unpaired)
FOLLOW_CONTRASTS = [("premise contradiction: follows, cited vs random", "premise:contradict_cited",
                     "premise:contradict_random"),
                    ("hypothesis contradiction: follows, cited vs random", "hypothesis:contradict_cited",
                     "hypothesis:contradict_random")]


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


def diff_ci(a, b, n_boot=1000, seed=0, alpha=0.05):
    """mean(a) - mean(b) for two INDEPENDENT samples, with a percentile bootstrap CI."""
    a, b = [float(x) for x in a], [float(x) for x in b]
    if not a or not b:
        return {"mean": None, "lo": None, "hi": None, "n": len(a), "n_control": len(b)}
    m = _mean(a) - _mean(b)
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        A, B = np.asarray(a), np.asarray(b)
        diffs = (A[rng.integers(0, len(A), size=(n_boot, len(A)))].mean(axis=1)
                 - B[rng.integers(0, len(B), size=(n_boot, len(B)))].mean(axis=1))
        lo, hi = (float(x) for x in np.quantile(diffs, [alpha / 2, 1 - alpha / 2]))
    except ImportError:
        rng = random.Random(seed)
        diffs = sorted(_mean([a[rng.randrange(len(a))] for _ in a]) - _mean([b[rng.randrange(len(b))] for _ in b])
                       for _ in range(200))
        lo, hi = diffs[int(alpha / 2 * 200)], diffs[int((1 - alpha / 2) * 200) - 1]
    return {"mean": m, "lo": lo, "hi": hi, "n": len(a), "n_control": len(b)}


def stratified_diff_ci(a, b, n_boot=1000, seed=0, alpha=0.05):
    """Difference of means a - b computed WITHIN strata and averaged with the stratum sizes of `a`.

    a, b: {stratum: [values]}. Only strata present in both are used. Used for contradiction edits,
    where cited and control edits differ in the mix of operations (align / swap_shared / swap) and
    the operations differ a lot in how often the model follows them: matching on the operation
    removes that difference from the cited-vs-control comparison. Bootstrap resamples each stratum.
    """
    strata = [s for s in a if a[s] and b.get(s)]
    n_a = sum(len(a[s]) for s in strata)
    out = {"mean": None, "lo": None, "hi": None, "n": n_a, "n_control": sum(len(b[s]) for s in strata),
           "strata": {s: {"n": len(a[s]), "n_control": len(b[s]), "diff": _mean(a[s]) - _mean(b[s])} for s in strata}}
    if not strata:
        return out
    w = {s: len(a[s]) / n_a for s in strata}
    out["mean"] = sum(w[s] * (_mean(a[s]) - _mean(b[s])) for s in strata)
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        tot = np.zeros(n_boot)
        for s in strata:
            A, B = np.asarray(a[s], dtype=float), np.asarray(b[s], dtype=float)
            tot += w[s] * (A[rng.integers(0, len(A), size=(n_boot, len(A)))].mean(axis=1)
                           - B[rng.integers(0, len(B), size=(n_boot, len(B)))].mean(axis=1))
        out["lo"], out["hi"] = (float(x) for x in np.quantile(tot, [alpha / 2, 1 - alpha / 2]))
    except ImportError:
        rng = random.Random(seed)
        tot = sorted(sum(w[s] * (_mean([a[s][rng.randrange(len(a[s]))] for _ in a[s]])
                                 - _mean([b[s][rng.randrange(len(b[s]))] for _ in b[s]])) for s in strata)
                     for _ in range(200))
        out["lo"], out["hi"] = tot[int(alpha / 2 * 200)], tot[int((1 - alpha / 2) * 200) - 1]
    return out


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


def _confident_oracle(r, key, min_conf):
    o = (r.get("oracle") or {}).get(key)
    if o is None or key not in r["variants"] or o["probs"][o["label"]] < min_conf:
        return None
    return o


def counterfactual_correctness(records, n_boot=1000, min_conf=0.9):
    out = {"removal": {}, "isolation": {}, "contradiction": {}, "contradiction_follow_contrasts": {},
           "contradiction_follow_contrasts_op_matched": {}, "oracle_min_conf": min_conf}
    for key in REMOVAL_KEYS:
        if not any(key in r["variants"] for r in records):
            continue
        out["removal"][key] = {}
        for g in ["all", "entailment", "contradiction"]:
            rs = [r for r in records if key in r["variants"] and r["original"]["label"] != "neutral" and _in_group(r, g)]
            new = [(r["original"]["label"], r["variants"][key]["label"]) for r in rs]
            out["removal"][key][g] = {"strict_to_neutral": _ci_dict([float(b == "neutral") for _, b in new], n_boot=n_boot),
                                      "lenient_moves_away": _ci_dict([float(a != b) for a, b in new], n_boot=n_boot)}
    for key in ISOLATION_KEYS:
        if not any(key in r["variants"] for r in records):
            continue
        out["isolation"][key] = {g: {"kept": _ci_dict([float(r["variants"][key]["label"] == r["original"]["label"])
                                                       for r in records if key in r["variants"] and _in_group(r, g)],
                                                      n_boot=n_boot)} for g in GROUPS}

    follow_values, follow_by_op = {}, {}
    for key in CONTRADICTION_KEYS:
        if not any(key in r["variants"] for r in records):
            continue
        out["contradiction"][key] = {}
        for g in GROUPS:
            n_built, rows = 0, []
            for r in records:
                if key not in r["variants"] or not _in_group(r, g):
                    continue
                n_built += 1
                o = _confident_oracle(r, key, min_conf)
                if o is not None:
                    op = ((r.get("contradiction_edits") or {}).get(key) or {}).get("op")
                    rows.append((r["original"]["label"], o["label"], r["variants"][key]["label"], op))
            by_op = {}
            for y, o, m, op in rows:
                if o != y:
                    by_op.setdefault(op, []).append(float(m == o))
            follow_by_op[(key, g)] = by_op
            rows = [x[:3] for x in rows]
            change = [m == o for y, o, m in rows if o != y]
            stay = [m == y for y, o, m in rows if o == y]
            out["contradiction"][key][g] = {
                "n_edits": n_built,
                "n_confident": len(rows),
                "oracle_changes_label": _ci_dict([float(o != y) for y, o, _ in rows], n_boot=n_boot),
                "correct": _ci_dict([float(m == o) for _, o, m in rows], n_boot=n_boot),
                "follows_when_should_change": _ci_dict([float(x) for x in change], n_boot=n_boot),
                "stays_when_should_stay": _ci_dict([float(x) for x in stay], n_boot=n_boot),
            }
            follow_values[(key, g)] = change
    for name, a, b in FOLLOW_CONTRASTS:
        if (a, "all") in follow_values and (b, "all") in follow_values:
            out["contradiction_follow_contrasts"][name] = {
                g: diff_ci(follow_values[(a, g)], follow_values[(b, g)], n_boot=n_boot) for g in GROUPS}
            # same comparison within each edit operation (cited and control edits differ in their mix)
            out["contradiction_follow_contrasts_op_matched"][name] = {
                g: stratified_diff_ci(follow_by_op[(a, g)], follow_by_op[(b, g)], n_boot=n_boot) for g in GROUPS}
    return out


# ---------------------------------------------------------------- explanation-evidence alignment
ALIGN_MEASURES = ["label_changed", "explanation_changed", "consistent", "cites_absent", "mentions_new",
                  "argues_expected", "aligned"]
# cited vs control: share of aligned regenerated explanations, on edits the oracle says should change the label
ALIGN_CONTRASTS = [("premise contradiction: aligned, cited vs random", "premise:contradict_cited",
                    "premise:contradict_random"),
                   ("hypothesis contradiction: aligned, cited vs random", "hypothesis:contradict_cited",
                    "hypothesis:contradict_random")]


def explanation_alignment(records, n_boot=1000, min_conf=0.9):
    """Summary of the regenerated explanations (alignment.py), or None if none were generated."""
    from alignment import features
    keys = sorted({k for r in records for k in (r.get("variant_expl") or {})})
    if not keys:
        return None
    feats = {k: [(r, f) for r in records for f in [features(r, k, min_conf)] if f is not None] for k in keys}

    def measures(fs):
        d = {"n": len(fs)}
        for m in ALIGN_MEASURES:
            vals = [float(f[m]) for f in fs if m in f]
            if vals:
                d[m] = _ci_dict(vals, n_boot=n_boot)
        return d

    out = {"n_examples": sum(1 for r in records if r.get("variant_expl")), "min_conf": min_conf,
           "variants": {}, "contrasts": {}, "contrasts_op_matched": {}}
    for k, rf in feats.items():
        fs = [f for _, f in rf]
        d = {"family": fs[0]["family"] if fs else None,
             "all": measures(fs),
             "label_changed": measures([f for f in fs if f["label_changed"]]),
             "label_kept": measures([f for f in fs if not f["label_changed"]]),
             "aligned_by_predicted_label": {g: _ci_dict([float(f["aligned"]) for r, f in rf if r["original"]["label"] == g],
                                                        n_boot=n_boot) for g in LABELS}}
        if d["family"] == "contradiction":
            d["should_change"] = measures([f for f in fs if f.get("should_change")])
            d["should_change_by_op"] = {}
            for op in sorted({f["op"] for f in fs}):
                d["should_change_by_op"][op] = measures([f for f in fs if f.get("should_change") and f["op"] == op])
        out["variants"][k] = d

    def should_change(k):
        return [f for _, f in feats.get(k, []) if f.get("should_change")]

    for name, a, b in ALIGN_CONTRASTS:
        fa, fb = should_change(a), should_change(b)
        if not fa or not fb:
            continue
        out["contrasts"][name] = diff_ci([float(f["aligned"]) for f in fa], [float(f["aligned"]) for f in fb],
                                         n_boot=n_boot)
        by_op = lambda fs: {op: [float(f["aligned"]) for f in fs if f["op"] == op] for op in {f["op"] for f in fs}}
        out["contrasts_op_matched"][name] = stratified_diff_ci(by_op(fa), by_op(fb), n_boot=n_boot)
    return out


def oracle_summary(records, min_conf=0.9):
    rs = [r for r in records if (r.get("oracle") or {}).get("original")]
    if not rs:
        return None
    conf_ok = [r for r in rs if r["oracle"]["original"]["probs"][r["oracle"]["original"]["label"]] >= min_conf]
    return {
        "n": len(rs),
        "accuracy_vs_gold_on_original_pairs": _mean([float(r["oracle"]["original"]["label"] == r["label"]) for r in rs]),
        "accuracy_vs_gold_when_confident": _mean([float(r["oracle"]["original"]["label"] == r["label"]) for r in conf_ok]),
        "share_confident": len(conf_ok) / len(rs),
        "agreement_with_model_on_original_pairs": _mean([float(r["oracle"]["original"]["label"] == r["original"]["label"])
                                                         for r in rs]),
    }


def summarise(records, n_boot=1000, oracle_min_conf=0.9):
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
        "counterfactual_correctness": counterfactual_correctness(records, n_boot, oracle_min_conf),
        "oracle": oracle_summary(records, oracle_min_conf),
        "explanation_alignment": explanation_alignment(records, n_boot, oracle_min_conf),
    }
