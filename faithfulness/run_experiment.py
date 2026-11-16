"""Faithfulness experiment on e-SNLI (resumable; safe to re-run after a Colab disconnect).

  1. model predicts label + explanation on the original pairs        -> originals.jsonl (cache)
  2. find the words its explanation cites (premise AND hypothesis)
  3. build counterfactual variants + matched controls + gold variants
     (removal, isolation, and evidence contradiction: one evidence word replaced by a conflicting one)
  4. model predicts the label on every variant (fast, no generation)   -> variant_preds.jsonl (cache)
  4b. an independent NLI oracle labels the original pairs and every contradiction edit:
      the oracle's label is the EXPECTED label of the edit             -> <results>/oracle_cache/*.jsonl
                                                                          (shared by all models)
  5. metrics with bootstrap CIs                                       -> summary.json
     plus records.jsonl (everything), per_example.csv (for reading),
     contradiction_edits.csv (every edit, for checking) and
     spotcheck_contradiction.csv (150 edits to label by hand; written once, never overwritten)

Re-running with the same --out_dir reuses every cached prediction, so adding the contradiction
intervention to an existing run only computes the new edits.

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
from interventions import build_contradiction_variants, build_variants
from metrics import summarise
from stance import stance
from swaps import Swapper

SIDES = ("premise", "hypothesis")
CONTRA_EDIT_FIELDS = ("op", "index", "target", "substitute")


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
    ap.add_argument("--no_contradiction", action="store_true", help="skip the evidence-contradiction intervention")
    ap.add_argument("--oracle", default=None, help="HF 3-way NLI model that labels contradiction edits "
                    "(default: predictors.DEFAULT_ORACLE)")
    ap.add_argument("--oracle_cache", default=None, help="oracle cache file (default: <parent of out_dir>/"
                    "oracle_cache/<oracle>.jsonl, shared by all model runs)")
    ap.add_argument("--oracle_batch_size", type=int, default=64)
    ap.add_argument("--oracle_min_conf", type=float, default=0.9,
                    help="only edits whose oracle label has at least this probability are scored")
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

    esnli_path = download_esnli(args.split, args.data_dir)
    examples = load_esnli(esnli_path, n=args.n, seed=args.seed)
    print(f"Loaded {len(examples)} e-SNLI {args.split} examples")
    swapper = None
    if not args.no_contradiction:
        # vocabulary from the WHOLE split, so the substitutions do not depend on --n
        swapper = Swapper([e[s] for e in load_esnli(esnli_path) for s in SIDES])

    # 1) original predictions with explanations
    originals = _run_cached(os.path.join(args.out_dir, "originals.jsonl"),
                            [(e["id"], (e["premise"], e["hypothesis"])) for e in examples],
                            predictor.predict, args.chunk, "originals (generate)")

    # 2-3) cited evidence and variants
    records, jobs = [], []
    for ex in examples:
        orig = originals[ex["id"]]
        cited = {s: cited_indices(ex[s], orig["explanation"]) for s in SIDES}
        variants, contra = {}, {}
        for s in SIDES:
            variants.update(build_variants(ex, cited[s], side=s, seed=args.seed))
            if swapper is not None:
                contra.update(build_contradiction_variants(ex, cited[s], swapper, side=s, seed=args.seed))
        variants.update({k: (e["premise"], e["hypothesis"]) for k, e in contra.items()})
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
            "contradiction_edits": {k: {f: e[f] for f in CONTRA_EDIT_FIELDS} for k, e in contra.items()},
            "variants": {},
        })

    # 4) label predictions on all variants
    vpreds = _run_cached(os.path.join(args.out_dir, "variant_preds.jsonl"), jobs,
                         predictor.predict_labels, args.chunk * 4, "variants (label only)")
    by_id = {r["id"]: r for r in records}
    for key, _ in jobs:
        ex_id, vname = key.split("||", 1)
        by_id[ex_id]["variants"][vname] = vpreds[key]

    # 4b) oracle: expected labels for the contradiction edits (+ original pairs, to report its accuracy)
    if swapper is not None:
        del predictor
        _free_gpu()
        run_oracle(records, args)

    # 5) outputs
    with open(os.path.join(args.out_dir, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    write_per_example_csv(records, os.path.join(args.out_dir, "per_example.csv"))
    if swapper is not None:
        write_contradiction_csvs(records, args.out_dir, args.oracle_min_conf, args.seed)

    print("Computing metrics (bootstrap)...")
    summary = summarise(records, n_boot=args.n_boot, oracle_min_conf=args.oracle_min_conf)
    summary["config"] = vars(args)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    print(f"\nWrote {args.out_dir}/  (records.jsonl, per_example.csv, summary.json)")
    print(f"Next: python faithfulness/plots.py --run_dir {args.out_dir}")


def _free_gpu():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _text_key(p, h):
    return f"{p}␟{h}"  # oracle cache is keyed by the text itself, so all model runs share it


def run_oracle(records, args):
    from predictors import DEFAULT_ORACLE, DummyOracle, NLIOracle
    name = args.oracle or DEFAULT_ORACLE
    if args.dry_run:
        oracle, cache = DummyOracle(), os.path.join(args.out_dir, "oracle_cache_DRYRUN.jsonl")
    else:
        cache = args.oracle_cache or os.path.join(os.path.dirname(os.path.abspath(args.out_dir)), "oracle_cache",
                                                  name.replace("/", "__") + ".jsonl")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        print(f"Loading NLI oracle {name}")
        oracle = NLIOracle(name, batch_size=args.oracle_batch_size)
    items = {}
    for r in records:
        items[_text_key(r["premise"], r["hypothesis"])] = (r["premise"], r["hypothesis"])
        for k in r["contradiction_edits"]:
            v = r["variant_inputs"][k]
            items[_text_key(v["premise"], v["hypothesis"])] = (v["premise"], v["hypothesis"])
    preds = _run_cached(cache, list(items.items()), oracle.predict, args.chunk, "oracle (expected labels)")
    for r in records:
        r["oracle"] = {"original": preds[_text_key(r["premise"], r["hypothesis"])]}
        for k in r["contradiction_edits"]:
            v = r["variant_inputs"][k]
            r["oracle"][k] = preds[_text_key(v["premise"], v["hypothesis"])]
    del oracle
    _free_gpu()


def write_contradiction_csvs(records, out_dir, min_conf, seed):
    """contradiction_edits.csv: every edit. spotcheck_contradiction.csv: a fixed random sample of
    confident edits with empty columns for a human check (never overwritten once it exists)."""
    import random
    header = ["id", "gold", "model_pred", "explanation", "variant", "op", "target", "substitute",
              "premise", "hypothesis", "edited_premise", "edited_hypothesis",
              "oracle_label", "oracle_conf", "oracle_changes_label", "model_pred_on_edit", "model_follows_oracle"]
    rows = []
    for r in records:
        y = r["original"]["label"]
        for k, e in r["contradiction_edits"].items():
            v, o, m = r["variant_inputs"][k], r["oracle"][k], r["variants"][k]["label"]
            rows.append([r["id"], r["label"], y, r["original"]["explanation"], k, e["op"], e["target"],
                         e["substitute"], r["premise"], r["hypothesis"], v["premise"], v["hypothesis"],
                         o["label"], round(o["probs"][o["label"]], 3), o["label"] != y, m, m == o["label"]])
    with open(os.path.join(out_dir, "contradiction_edits.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    spot = os.path.join(out_dir, "spotcheck_contradiction.csv")
    if os.path.exists(spot):
        return
    conf_i, var_i = header.index("oracle_conf"), header.index("variant")
    pool = [x for x in rows if x[conf_i] >= min_conf and not x[var_i].endswith("_random")]
    sample = random.Random(seed).sample(pool, min(150, len(pool)))
    keep = ["id", "variant", "op", "target", "substitute", "premise", "hypothesis", "edited_premise",
            "edited_hypothesis", "oracle_label"]
    with open(spot, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keep + ["edit_is_natural (y/n)", "your_label (entailment/neutral/contradiction)", "notes"])
        for x in sample:
            w.writerow([x[header.index(c)] for c in keep] + ["", "", ""])


def write_per_example_csv(records, path):
    cols = ["premise:remove_cited", "premise:remove_random", "premise:remove_overlap", "premise:isolate_cited",
            "hypothesis:remove_cited", "hypothesis:remove_random",
            "premise:contradict_cited", "hypothesis:contradict_cited"]
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

    o = s.get("oracle")
    if o:
        print(f"\nOracle on original pairs: accuracy vs gold {o['accuracy_vs_gold_on_original_pairs']:.3f} "
              f"(confident: {o['accuracy_vs_gold_when_confident']:.3f}, {o['share_confident']:.0%} of pairs); "
              f"agreement with the model {o['agreement_with_model_on_original_pairs']:.3f}")
    cc = s["counterfactual_correctness"]
    if cc["contradiction"]:
        print(f"\nEvidence contradiction (oracle p >= {cc['oracle_min_conf']}):")
        print(f"  {'variant':<30}{'edits':>7}{'scored':>8}{'should change':>15}{'follows':>10}{'stays':>8}")
        for k, d in cc["contradiction"].items():
            a = d["all"]
            fmt = lambda x: "   -  " if x["mean"] is None else f"{x['mean']:.3f}"
            print(f"  {k:<30}{a['n_edits']:>7}{a['n_confident']:>8}{fmt(a['oracle_changes_label']):>15}"
                  f"{fmt(a['follows_when_should_change']):>10}{fmt(a['stays_when_should_stay']):>8}")
        for name, d in cc["contradiction_follow_contrasts"].items():
            x = d["all"]
            if x["mean"] is not None:
                print(f"  {name}: {x['mean']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]")


if __name__ == "__main__":
    main()
