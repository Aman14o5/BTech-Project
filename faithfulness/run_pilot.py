"""Pilot faithfulness experiment on e-SNLI.

  1. load N e-SNLI test pairs (with human highlights)
  2. model predicts label + explanation on the original pairs
  3. find the words its explanation cites
  4. build counterfactual variants (remove / isolate cited, random controls, gold)
  5. model predicts again on every variant
  6. compute faithfulness metrics; write records, a per-example CSV and a summary

Run from the repo root:
  python faithfulness/run_pilot.py --model_path model/esnli/random/sub0/nt8 --n 300
  python faithfulness/run_pilot.py --dry_run --n 50        # no GPU/model; tests the pipeline
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import download_esnli, load_esnli
from evidence import cited_indices, overlap_prf
from interventions import build_variants
from metrics import summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", help="directory of a fine-tuned T5 checkpoint (HF format)")
    ap.add_argument("--dry_run", action="store_true", help="use a rule-based dummy predictor (pipeline test only)")
    ap.add_argument("--n", type=int, default=300, help="number of e-SNLI test examples")
    ap.add_argument("--side", default="premise", choices=["premise", "hypothesis"], help="which sentence to intervene on")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--data_dir", default="faithfulness/data")
    ap.add_argument("--out_dir", default="faithfulness/results/pilot")
    args = ap.parse_args()

    if args.dry_run:
        from predictors import DummyPredictor
        predictor = DummyPredictor()
        args.out_dir = args.out_dir.rstrip("/") + "_DRYRUN"
    else:
        if not args.model_path:
            ap.error("--model_path is required unless --dry_run")
        from predictors import T5Predictor
        predictor = T5Predictor(args.model_path, batch_size=args.batch_size)
    os.makedirs(args.out_dir, exist_ok=True)

    examples = load_esnli(download_esnli("test", args.data_dir), n=args.n, seed=args.seed)
    print(f"Loaded {len(examples)} examples")

    # 1) original predictions
    originals = predictor.predict([(e["premise"], e["hypothesis"]) for e in examples])

    # 2) cited evidence + variants
    jobs = []  # (example index, variant name, pair)
    records = []
    for i, (ex, orig) in enumerate(zip(examples, originals)):
        cited = cited_indices(ex[args.side], orig["explanation"])
        variants = build_variants(ex, cited, side=args.side, seed=args.seed)
        jobs += [(i, name, pair) for name, pair in variants.items()]
        records.append({
            **ex,
            "side": args.side,
            "original": orig,
            "cited_idx": cited,
            "cited_words": [ex[args.side].split()[j] for j in cited],
            "cited_vs_gold": overlap_prf(cited, ex[f"gold_{args.side}_idx"]) if ex[f"gold_{args.side}_idx"] else None,
            "variant_inputs": {name: {"premise": p, "hypothesis": h} for name, (p, h) in variants.items()},
            "variants": {},
        })
    print(f"Built {len(jobs)} variants")

    # 3) predictions on variants (one batched pass)
    preds = predictor.predict([pair for _, _, pair in jobs])
    for (i, name, _), pred in zip(jobs, preds):
        records[i]["variants"][name] = pred

    # 4) outputs
    with open(os.path.join(args.out_dir, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with open(os.path.join(args.out_dir, "per_example.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "gold", "pred", "p_pred", "premise", "hypothesis", "explanation", "cited_words",
                    "pred_remove_cited", "drop_remove_cited", "pred_remove_random", "drop_remove_random",
                    "pred_isolate_cited", "suspect_unfaithful"])
        for r in records:
            y, p0 = r["original"]["label"], r["original"]["probs"][r["original"]["label"]]
            v = r["variants"]
            rc, rr, ic = v.get("remove_cited"), v.get("remove_random"), v.get("isolate_cited")
            # cited evidence removed, yet prediction and confidence essentially unchanged
            suspect = bool(rc and rc["label"] == y and p0 - rc["probs"][y] < 0.05)
            w.writerow([r["id"], r["label"], y, round(p0, 3), r["premise"], r["hypothesis"],
                        r["original"]["explanation"], " ".join(r["cited_words"]),
                        rc and rc["label"], rc and round(p0 - rc["probs"][y], 3),
                        rr and rr["label"], rr and round(p0 - rr["probs"][y], 3),
                        ic and ic["label"], suspect])

    summary = summarise(records)
    summary["config"] = vars(args)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)
    print(f"\nWrote {args.out_dir}/records.jsonl, per_example.csv, summary.json")


def _fmt(x):
    return "  -  " if x is None else f"{x:+.3f}" if isinstance(x, float) and x < 0 else f"{x:.3f}"


def print_summary(s):
    print(f"\nBaseline: acc={s['baseline_accuracy']:.3f}  macro-F1={s['baseline_macro_f1']:.3f}  (n={s['n_examples']})")
    print(f"Examples whose explanation cites input words: {s['frac_with_cited_evidence']:.1%}")
    if s["cited_vs_gold_f1"] is not None:
        print(f"Cited vs human highlights: P={s['cited_vs_gold_precision']:.3f} R={s['cited_vs_gold_recall']:.3f} F1={s['cited_vs_gold_f1']:.3f}")
    print(f"\n{'intervention':<16}{'n':>6}{'flip':>9}{'prob drop':>11}")
    for name, st in s["interventions"].items():
        print(f"{name:<16}{st['n']:>6}{_fmt(st['flip_rate']):>9}{_fmt(st['mean_prob_drop']):>11}")
    print("\nFaithfulness (cited minus random, same examples):")
    for scope, f in [("all", s["faithfulness"])] + [(l, d["faithfulness"]) for l, d in s["by_gold_label"].items()]:
        print(f"  {scope:<14} sensitivity={_fmt(f.get('evidence_sensitivity'))}  "
              f"sensitivity(prob)={_fmt(f.get('sensitivity_prob'))}  sufficiency_gain={_fmt(f.get('sufficiency_gain'))}  "
              f"n={f.get('removal_n_paired', 0)}")


if __name__ == "__main__":
    main()
