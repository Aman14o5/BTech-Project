"""Faithfulness metrics computed from per-example records.

For each intervention v and example x with original prediction y:
  flip(v)      = 1[pred(v) != y]
  drop(v)      = P(y | original) - P(y | v)        (confidence lost in the original label)
Aggregates:
  evidence_sensitivity      = flip_rate(remove_cited) - flip_rate(remove_random)
  sensitivity_prob          = mean_drop(remove_cited) - mean_drop(remove_random)
  sufficiency_gain          = keep_rate(isolate_cited) - keep_rate(isolate_random)
Positive values mean the cited evidence matters more than arbitrary words,
i.e. evidence that the explanation is faithful. Values near zero (or negative)
mean the explanation's evidence is no more important than random words.
"""
from collections import defaultdict

LABELS = ["entailment", "neutral", "contradiction"]


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


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def intervention_stats(records, subset=None):
    """records: list of dicts with 'original' and 'variants' (name -> prediction)."""
    flips, drops = defaultdict(list), defaultdict(list)
    for r in records:
        if subset and r["label"] != subset:
            continue
        y = r["original"]["label"]
        p0 = r["original"]["probs"][y]
        for name, pred in r["variants"].items():
            flips[name].append(pred["label"] != y)
            drops[name].append(p0 - pred["probs"][y])
    stats = {}
    for name in sorted(flips):
        stats[name] = {
            "n": len(flips[name]),
            "flip_rate": _mean(flips[name]),
            "keep_rate": 1 - _mean(flips[name]),
            "mean_prob_drop": _mean(drops[name]),
        }
    return stats


def paired_summary(records, subset=None):
    """Cited-vs-random comparisons on the *same* examples (only those with both variants)."""
    out = {}
    for kind, cited, rand in (("removal", "remove_cited", "remove_random"),
                              ("isolation", "isolate_cited", "isolate_random")):
        pairs = [r for r in records if (not subset or r["label"] == subset)
                 and cited in r["variants"] and rand in r["variants"]]
        if not pairs:
            continue
        sub = intervention_stats(pairs)
        c, rn = sub[cited], sub[rand]
        if kind == "removal":
            out["evidence_sensitivity"] = c["flip_rate"] - rn["flip_rate"]
            out["sensitivity_prob"] = c["mean_prob_drop"] - rn["mean_prob_drop"]
        else:
            out["sufficiency_gain"] = c["keep_rate"] - rn["keep_rate"]
        out[f"{kind}_n_paired"] = len(pairs)
    return out


def summarise(records):
    gold = [r["label"] for r in records]
    pred = [r["original"]["label"] for r in records]
    acc, f1 = accuracy_macro_f1(gold, pred)
    align = [r["cited_vs_gold"] for r in records if r.get("cited_vs_gold")]
    summary = {
        "n_examples": len(records),
        "baseline_accuracy": acc,
        "baseline_macro_f1": f1,
        "frac_with_cited_evidence": _mean([bool(r["cited_idx"]) for r in records]),
        "cited_vs_gold_f1": _mean([a["f1"] for a in align]),
        "cited_vs_gold_precision": _mean([a["precision"] for a in align]),
        "cited_vs_gold_recall": _mean([a["recall"] for a in align]),
        "interventions": intervention_stats(records),
        "faithfulness": paired_summary(records),
        "by_gold_label": {},
    }
    for l in LABELS:
        summary["by_gold_label"][l] = {
            "interventions": intervention_stats(records, subset=l),
            "faithfulness": paired_summary(records, subset=l),
        }
    return summary
