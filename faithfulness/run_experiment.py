"""Faithfulness experiment on e-SNLI (resumable; safe to re-run after a Colab disconnect).

  1. model predicts label + explanation on the original pairs        -> originals.jsonl (cache)
  2. find the words its explanation cites (premise AND hypothesis)
  3. build counterfactual variants + matched controls + gold variants
  4. model predicts the label on every variant (fast, no generation)   -> variant_preds.jsonl (cache)
  5. metrics with bootstrap CIs                                       -> summary.json
     plus records.jsonl (everything) and per_example.csv (for reading)

Run from the repo root:
  python faithfulness/run_experiment.py --model_path <ckpt> --n 2000 --out_dir <dir>
  python faithfulness/run_experiment.py --dry_run --n 300            # pipeline test, numbers meaningless
Then: python faithfulness/plots.py --run_dir <dir>
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import download_esnli, load_esnli
from evidence import cited_indices, overlap_prf
from interventions import build_variants
from metrics import summarise
from stance import stance

SIDES = ("premise", "hypothesis")


def _load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    cache[d["key"]] = d["pred"]
                except (json.JSONDecodeError, KeyError):
                    pass  # partially written last line after a crash
    return cache


def _run_cached(path, items, fn, chunk, desc):
    """items: list of (key, pair). Computes missing keys in chunks, appending to `path`."""
    cache = _load_cache(path)
    todo = [(k, p) for k, p in items if k not in cache]
    print(f"{desc}: {len(items) - len(todo)} cached, {len(todo)} to compute")
    t0 = time.time()
    with open(path, "a") as f:
        for s in range(0, len(todo), chunk):
            part = todo[s:s + chunk]
            for (k, _), pred in zip(part, fn([p for _, p in part])):
                cache[k] = pred
                f.write(json.dumps({"key": k, "pred": pred}) + "\n")
            f.flush()
            done = s + len(part)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {desc}: {done}/{len(todo)}  ({rate:.1f}/s, ~{(len(todo) - done) / max(rate, 1e-6) / 60:.1f} min left)")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", help="fine-tuned T5 checkpoint dir (HF format)")
    ap.add_argument("--dry_run", action="store_true", help="rule-based dummy predictor (pipeline test only)")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--n", type=int, default=None, help="number of examples (default: whole split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32, help="batch size for generation")
    ap.add_argument("--label_batch_size", type=int, default=128, help="batch size for label-only scoring")
    ap.add_argument("--chunk", type=int, default=1024, help="examples per cache write")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--data_dir", default="faithfulness/data")
    ap.add_argument("--out_dir", default="faithfulness/results/run")
    args = ap.parse_args()

    if args.dry_run:
        from predictors import DummyPredictor
        predictor = DummyPredictor()
        args.out_dir = args.out_dir.rstrip("/") + "_DRYRUN"
    else:
        if not args.model_path:
            ap.error("--model_path is required unless --dry_run")
        from predictors import T5Predictor
        predictor = T5Predictor(args.model_path, batch_size=args.batch_size, label_batch_size=args.label_batch_size)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    examples = load_esnli(download_esnli(args.split, args.data_dir), n=args.n, seed=args.seed)
    print(f"Loaded {len(examples)} e-SNLI {args.split} examples")

    # 1) original predictions with explanations
    originals = _run_cached(os.path.join(args.out_dir, "originals.jsonl"),
                            [(e["id"], (e["premise"], e["hypothesis"])) for e in examples],
                            predictor.predict, args.chunk, "originals (generate)")

    # 2-3) cited evidence and variants
    records, jobs = [], []
    for ex in examples:
        orig = originals[ex["id"]]
        cited = {s: cited_indices(ex[s], orig["explanation"]) for s in SIDES}
        variants = {}
        for s in SIDES:
            variants.update(build_variants(ex, cited[s], side=s, seed=args.seed))
        jobs += [(f"{ex['id']}||{k}", pair) for k, pair in variants.items()]
        records.append({
            **ex,
            "original": orig,
            "stance": stance(orig["explanation"]),
            "human_stance": stance(ex["explanation"]),
            "cited": cited,
            "cited_words": {s: [ex[s].split()[j] for j in cited[s]] for s in SIDES},
            "cited_vs_gold": {s: overlap_prf(cited[s], ex[f"gold_{s}_idx"]) if ex[f"gold_{s}_idx"] else None
                              for s in SIDES},
            "variant_inputs": {k: {"premise": p, "hypothesis": h} for k, (p, h) in variants.items()},
            "variants": {},
        })

    # 4) label predictions on all variants
    vpreds = _run_cached(os.path.join(args.out_dir, "variant_preds.jsonl"), jobs,
                         predictor.predict_labels, args.chunk * 4, "variants (label only)")
    by_id = {r["id"]: r for r in records}
    for key, _ in jobs:
        ex_id, vname = key.split("||", 1)
        by_id[ex_id]["variants"][vname] = vpreds[key]

    # 5) outputs
    with open(os.path.join(args.out_dir, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    write_per_example_csv(records, os.path.join(args.out_dir, "per_example.csv"))

    print("Computing metrics (bootstrap)...")
    summary = summarise(records, n_boot=args.n_boot)
    summary["config"] = vars(args)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    print(f"\nWrote {args.out_dir}/  (records.jsonl, per_example.csv, summary.json)")
    print(f"Next: python faithfulness/plots.py --run_dir {args.out_dir}")


def write_per_example_csv(records, path):
    cols = ["premise:remove_cited", "premise:remove_random", "premise:remove_overlap", "premise:isolate_cited",
            "hypothesis:remove_cited", "hypothesis:remove_random"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "gold", "pred", "p_pred", "premise", "hypothesis", "explanation", "explanation_stance",
                    "consistent", "cited_premise", "cited_hypothesis"]
                   + [f"{c}{suf}" for c in cols for suf in ("_pred", "_drop")]
                   + ["suspect_unfaithful"])
        for r in records:
            y = r["original"]["label"]
            p0 = r["original"]["probs"][y]
            row = [r["id"], r["label"], y, round(p0, 3), r["premise"], r["hypothesis"], r["original"]["explanation"],
                   r["stance"], r["stance"] == y, " ".join(r["cited_words"]["premise"]),
                   " ".join(r["cited_words"]["hypothesis"])]
            for c in cols:
                v = r["variants"].get(c)
                row += [v["label"], round(p0 - v["probs"][y], 3)] if v else ["", ""]
            rc = r["variants"].get("premise:remove_cited")
            # cited premise evidence removed, yet prediction and confidence essentially unchanged,
            # for a decision that should depend on premise evidence (entailment/contradiction)
            suspect = bool(rc and y != "neutral" and rc["label"] == y and p0 - rc["probs"][y] < 0.05)
            w.writerow(row + [suspect or (r["stance"] != y)])


def _f(x):
    return "   -  " if x is None else f"{x:+.3f}"


def print_summary(s):
    b = s["baseline"]
    print(f"\nBaseline: acc={b['accuracy']:.3f}  macro-F1={b['macro_f1']:.3f}  n={s['n_examples']}  "
          f"pred counts={b['pred_label_counts']}")
    print(f"Explanation cites premise words: {s['cites_input']['premise']:.1%}   "
          f"hypothesis words: {s['cites_input']['hypothesis']:.1%}")
    c = s["consistency"]
    print(f"Label-explanation consistency (model): {c['model']['all']['mean']:.3f} "
          f"[{c['model']['all']['lo']:.3f}, {c['model']['all']['hi']:.3f}]   "
          f"stance-rule accuracy on human explanations: {c['human_reference_accuracy']:.3f}")
    print("\nContrasts (cited minus control, flip-rate difference, 95% CI):")
    print(f"  {'contrast':<36}" + "".join(f"{g:>30}" for g in ("all", "entailment", "neutral", "contradiction")))
    for name, d in s["contrasts"].items():
        cells = []
        for g in ("all", "entailment", "neutral", "contradiction"):
            x = d[g]["flip_diff"]
            cells.append("        -" if x["mean"] is None else f"{_f(x['mean'])} [{_f(x['lo'])},{_f(x['hi'])}] n={x['n']}  ")
        print(f"  {name:<36}" + "".join(f"{c:>30}" for c in cells))


if __name__ == "__main__":
    main()
