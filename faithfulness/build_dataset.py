"""Build the model-independent counterfactual dataset from e-SNLI human highlights.

For every e-SNLI pair, the words human annotators highlighted as evidence are
removed or isolated, and a matched random control is created. These variants do
not depend on any model, so the same file can be used to test every model and
can be released as a dataset.

Output (in --out_dir):
  esnli_cf_gold_<split>.jsonl   one line per pair: original pair, gold label,
                                human explanation + its stance, highlight indices,
                                and {variant_key: {premise, hypothesis}}
  esnli_cf_gold_<split>_stats.json   counts per variant and label, highlight
                                statistics, and validation of the stance rules
                                on the human explanations

  python faithfulness/build_dataset.py --split test
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import download_esnli, load_esnli, LABELS
from interventions import build_variants
from stance import stance, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="faithfulness/data")
    ap.add_argument("--out_dir", default="faithfulness/data")
    args = ap.parse_args()

    examples = load_esnli(download_esnli(args.split, args.data_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"esnli_cf_gold_{args.split}.jsonl")

    counts = defaultdict(Counter)
    n_highlight = defaultdict(list)
    with open(out_path, "w") as f:
        for ex in examples:
            variants = {}
            for side in ("premise", "hypothesis"):
                # cited=[] -> only the gold-highlight variants are built
                variants.update(build_variants(ex, [], side=side, seed=args.seed))
                n_highlight[(side, ex["label"])].append(len(ex[f"gold_{side}_idx"]))
            for k in variants:
                counts[k][ex["label"]] += 1
            f.write(json.dumps({
                "id": ex["id"], "label": ex["label"], "premise": ex["premise"], "hypothesis": ex["hypothesis"],
                "human_explanation": ex["explanation"], "human_explanation_stance": stance(ex["explanation"]),
                "gold_premise_idx": ex["gold_premise_idx"], "gold_hypothesis_idx": ex["gold_hypothesis_idx"],
                "variants": {k: {"premise": p, "hypothesis": h} for k, (p, h) in variants.items()},
            }) + "\n")

    acc, conf = validate(examples)
    stats = {
        "split": args.split,
        "n_pairs": len(examples),
        "label_counts": dict(Counter(e["label"] for e in examples)),
        "variants_per_label": {k: dict(v) for k, v in sorted(counts.items())},
        "n_variants_total": sum(sum(v.values()) for v in counts.values()),
        "mean_highlighted_words": {f"{s}/{l}": round(sum(v) / len(v), 2) for (s, l), v in sorted(n_highlight.items())},
        "frac_with_highlight": {f"{s}/{l}": round(sum(x > 0 for x in v) / len(v), 3)
                                for (s, l), v in sorted(n_highlight.items())},
        "stance_rules_validation": {"accuracy_on_human_explanations": round(acc, 4), "confusion_gold_x_stance": conf},
    }
    with open(out_path.replace(".jsonl", "_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {out_path}  ({stats['n_pairs']} pairs, {stats['n_variants_total']} counterfactual variants)")
    print(f"Stance rules vs human explanations: accuracy {acc:.3f}")
    for k, v in stats["variants_per_label"].items():
        print(f"  {k:<36} " + "  ".join(f"{l[:4]}={v.get(l, 0)}" for l in LABELS))


if __name__ == "__main__":
    main()
