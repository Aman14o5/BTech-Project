"""Analyse the human ratings of explanation quality (plan Table 2 "Human rating"; RQ2, RQ3).

Inputs (from build_rating_sheets.py):
  human_rating/rating_sheet_rater{1,2,3}.xlsx   (or .numbers if a rater filled it in Apple Numbers)
  human_rating/rating_key.csv                   item -> model, run, pair id, gold label, flags
  btp_results/<run>/per_example.csv             removal / isolation results per pair
  btp_results/<run>/variant_explanations.csv    regenerated-explanation alignment per pair

Rating scale (Yang et al.): Yes / Weakly Yes / Weakly No / No -> 1, 2/3, 1/3, 0.

Steps
  1. validity: every rater rated all items with a valid value
  2. inter-rater agreement: Krippendorff's alpha (ordinal and interval), pairwise Spearman, exact agreement
  3. per-item quality = mean of the three scores
  4. RQ2: Spearman rho between quality and each per-example faithfulness measure, pooled and per model,
     plus a model-level table (n=4, descriptive only)
  5. RQ3: plausible-but-unfaithful = quality >= 2/3 but failing a faithfulness test; rate by model and
     label, premise length, shortcomings, examples

Writes to --out_dir: ratings_merged.csv, rating_analysis.json, and prints a readable report.

Run from the repo root (needs numpy, scipy, openpyxl; numbers-parser only for .numbers sheets):
  python faithfulness/analyse_ratings.py --ratings human_rating --results btp_results --out_dir btp_results/human_rating
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

SCORE = {"Yes": 1.0, "Weakly Yes": 2 / 3, "Weakly No": 1 / 3, "No": 0.0}
SHORTCOMINGS = ["Does not make sense", "Insufficient", "Irrelevant", "Only repeats", "Hallucinated"]
MODELS = ["random_nt8", "fastvotek_nt8", "random_nt128", "fastvotek_nt128"]
PLAUSIBLE = 2 / 3 - 1e-9


# ---------------------------------------------------------------- reading the sheets
def _rows_xlsx(path):
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["Ratings"]
    return [list(r) for r in ws.iter_rows(values_only=True)]


def _rows_numbers(path):
    from numbers_parser import Document
    for sheet in Document(path).sheets:
        if sheet.name == "Ratings":
            return [[c.value for c in r] for r in sheet.tables[0].rows()]
    raise ValueError(f"no Ratings sheet in {path}")


def _parse(rows):
    hdr = [str(h or "").strip() for h in rows[0]]
    ci, cq = hdr.index("Item"), next(i for i, h in enumerate(hdr) if h.startswith("Q1"))
    cs = {s: hdr.index(s) for s in SHORTCOMINGS}
    cn = hdr.index("Notes") if "Notes" in hdr else None
    out = {}
    for r in rows[1:]:
        if not r or not r[ci]:
            continue
        v = str(r[cq]).strip() if r[cq] not in (None, "") else None
        out[str(r[ci]).strip()] = {
            "rating": v,
            "short": {s: r[c] not in (None, "", False) for s, c in cs.items()},
            "notes": (r[cn] or "") if cn is not None and cn < len(r) else "",
        }
    return out


def load_rater(ratings_dir, k):
    """Use whichever of the .xlsx / .numbers sheets has more ratings filled in."""
    best = None
    for ext, reader in ((".xlsx", _rows_xlsx), (".numbers", _rows_numbers)):
        p = os.path.join(ratings_dir, f"rating_sheet_rater{k}{ext}")
        if not os.path.exists(p):
            continue
        try:
            d = _parse(reader(p))
        except Exception as e:  # noqa: BLE001 - report and try the other file
            print(f"  could not read {p}: {e}")
            continue
        n = sum(x["rating"] is not None for x in d.values())
        if best is None or n > best[2]:
            best = (p, d, n)
    if best is None:
        raise FileNotFoundError(f"no sheet for rater {k}")
    return best


# ---------------------------------------------------------------- agreement
def krippendorff_alpha(data, metric="interval"):
    """data: raters x items array (np.nan = missing); values are ordered numbers."""
    data = np.asarray(data, float)
    items = [col[~np.isnan(col)] for col in data.T]
    items = [v for v in items if len(v) >= 2]
    values = np.unique(np.concatenate(items))
    idx = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))  # coincidence matrix
    for v in items:
        m = len(v)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[idx[v[a]], idx[v[b]]] += 1 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    if metric == "interval":
        d = (values[:, None] - values[None, :]) ** 2
    elif metric == "ordinal":
        d = np.zeros_like(o)
        for c in range(len(values)):
            for k in range(len(values)):
                lo, hi = min(c, k), max(c, k)
                d[c, k] = (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
    else:  # nominal
        d = 1 - np.eye(len(values))
    d_o = (o * d).sum() / n
    d_e = (np.outer(n_c, n_c) * d).sum() / (n * (n - 1))
    return 1 - d_o / d_e


def bootstrap_alpha(mat, metric, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    m = mat.shape[1]
    vals = [krippendorff_alpha(mat[:, rng.integers(0, m, m)], metric) for _ in range(n_boot)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------------------------------------------------------------- faithfulness measures
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _b(x):
    return None if x in ("", None) else str(x) == "True"


def load_faithfulness(results_dir, key_rows):
    """Per (run, pair id): the per-example measures and the alignment results."""
    want = defaultdict(set)
    for r in key_rows:
        want[r["run"]].add(r["id"])
    out = {}
    for run, ids in want.items():
        with open(os.path.join(results_dir, run, "per_example.csv")) as f:
            for r in csv.DictReader(f):
                if r["id"] in ids:
                    out[(run, r["id"])] = {"per_example": r, "align": {}}
        p = os.path.join(results_dir, run, "variant_explanations.csv")
        with open(p) as f:
            for r in csv.DictReader(f):
                k = (run, r["id"])
                if k in out:
                    out[k]["align"][r["variant"]] = r
    return out


def measures(fa, pred):
    """Per-example faithfulness measures. Higher = more faithful for every continuous one."""
    pe, al = fa["per_example"], fa["align"]
    m = {}
    m["consistent"] = _b(pe["consistent"])  # explanation stance agrees with the predicted label
    m["suspect_unfaithful"] = _b(pe["suspect_unfaithful"])
    rc, rr = _f(pe["premise:remove_cited_drop"]), _f(pe["premise:remove_random_drop"])
    m["remove_cited_drop"] = rc  # confidence drop when the cited premise words are deleted
    m["cited_minus_random_drop"] = rc - rr if rc is not None and rr is not None else None
    m["remove_cited_flip"] = (pe["premise:remove_cited_pred"] != pred) if pe["premise:remove_cited_pred"] else None
    ic = pe["premise:isolate_cited_pred"]
    m["isolate_cited_keeps"] = (ic == pred) if ic else None  # cited words alone keep the label
    # removal/isolation tests only make sense for decisions that rest on premise evidence
    if pred == "neutral":
        for k in ("remove_cited_drop", "cited_minus_random_drop", "remove_cited_flip", "isolate_cited_keeps"):
            m[k] = None
    aligned = [_b(r["aligned"]) for r in al.values() if _b(r["aligned"]) is not None]
    m["alignment_rate"] = float(np.mean(aligned)) if aligned else None  # share of regenerated expl. aligned
    rcv = al.get("premise:remove_cited")
    m["remove_cited_expl_consistent"] = _b(rcv["consistent"]) if rcv else None
    pcc = al.get("premise:contradict_cited")
    m["contradict_cited_aligned"] = _b(pcc["aligned"]) if pcc else None
    return m


MEASURES = ["consistent", "suspect_unfaithful", "remove_cited_drop", "cited_minus_random_drop",
            "remove_cited_flip", "isolate_cited_keeps", "alignment_rate", "remove_cited_expl_consistent",
            "contradict_cited_aligned"]
# a pair "fails the tests" (RQ3) if any of these is True
FAILS = {
    "stance_disagrees": lambda m: m["consistent"] is False,
    "removal_insensitive": lambda m: m["remove_cited_drop"] is not None and m["remove_cited_flip"] is False
    and m["remove_cited_drop"] < 0.05,
    "isolation_loses_label": lambda m: m["isolate_cited_keeps"] is False,
    "contradiction_misaligned": lambda m: m["contradict_cited_aligned"] is False,
}


def spearman(x, y, n_boot=1000, seed=0):
    pairs = [(a, float(b)) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 8:
        return {"n": len(pairs)}
    a, b = map(np.array, zip(*pairs))
    if a.std() == 0 or b.std() == 0:
        return {"n": len(pairs)}
    rho, p = spearmanr(a, b)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        if a[i].std() > 0 and b[i].std() > 0:
            boots.append(spearmanr(a[i], b[i])[0])
    return {"n": len(pairs), "rho": float(rho), "p": float(p),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="human_rating")
    ap.add_argument("--results", default="btp_results")
    ap.add_argument("--out_dir", default="btp_results/human_rating")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    with open(os.path.join(a.ratings, "rating_key.csv")) as f:
        key = list(csv.DictReader(f))
    items = [r["Item"] if "Item" in r else r["item"] for r in key]

    # 1. validity
    raters, report = [], {"validity": {}}
    print("== 1. Validity")
    for k in (1, 2, 3):
        path, d, n = load_rater(a.ratings, k)
        bad = [i for i in items if d.get(i, {}).get("rating") not in SCORE]
        nshort = sum(any(d[i]["short"].values()) for i in items if i in d)
        report["validity"][f"rater{k}"] = {"file": os.path.basename(path), "rated": n, "missing_or_invalid": len(bad),
                                           "rows_with_shortcomings": nshort}
        print(f"  rater {k}: {os.path.basename(path):28s} rated {n}/{len(items)}  invalid/missing {len(bad)}"
              f"  rows with shortcomings {nshort}")
        raters.append(d)
    mat = np.array([[SCORE.get(d.get(i, {}).get("rating"), np.nan) for i in items] for d in raters])

    # 2. agreement
    print("\n== 2. Inter-rater agreement (3 raters, 200 items)")
    ag = {}
    for metric in ("ordinal", "interval", "nominal"):
        ag[f"alpha_{metric}"] = float(krippendorff_alpha(mat, metric))
    ag["alpha_interval_ci"] = bootstrap_alpha(mat, "interval")
    ag["pairwise"] = {}
    for i in range(3):
        for j in range(i + 1, 3):
            rho = spearmanr(mat[i], mat[j])[0]
            exact = float(np.mean(mat[i] == mat[j]))
            within1 = float(np.mean(np.abs(mat[i] - mat[j]) < 0.34))
            ag["pairwise"][f"r{i + 1}-r{j + 1}"] = {"spearman": float(rho), "exact": exact, "within_one_step": within1}
    binm = (mat >= PLAUSIBLE).astype(float)  # Yes/Weakly Yes vs Weakly No/No
    ag["alpha_binary_accept"] = float(krippendorff_alpha(binm, "nominal"))
    ag["rater_means"] = [float(x) for x in mat.mean(1)]
    ag["rater_accept_rate"] = [float(x) for x in binm.mean(1)]
    report["agreement"] = ag
    print(f"  Krippendorff alpha  ordinal {ag['alpha_ordinal']:.3f}  interval {ag['alpha_interval']:.3f} "
          f"[{ag['alpha_interval_ci'][0]:.3f}, {ag['alpha_interval_ci'][1]:.3f}]  nominal {ag['alpha_nominal']:.3f}"
          f"  binary accept {ag['alpha_binary_accept']:.3f}")
    for k, v in ag["pairwise"].items():
        print(f"  {k}: spearman {v['spearman']:.3f}  exact {v['exact']:.0%}  within one step {v['within_one_step']:.0%}")
    print("  mean score per rater:", " ".join(f"{x:.3f}" for x in ag["rater_means"]),
          "  accept rate:", " ".join(f"{x:.0%}" for x in ag["rater_accept_rate"]))

    # 3. per-item quality + faithfulness
    fa = load_faithfulness(a.results, key)
    rows = []
    for col, r in enumerate(key):
        q = float(np.nanmean(mat[:, col]))
        f = fa.get((r["run"], r["id"]))
        m = measures(f, r["relationship"]) if f else {k: None for k in MEASURES}
        fails = {k: fn(m) for k, fn in FAILS.items()} if f else {}
        short = {s: sum(d[items[col]]["short"][s] for d in raters if items[col] in d) for s in SHORTCOMINGS}
        rows.append({"item": items[col], "model": r["model"], "id": r["id"], "gold": r["gold"],
                     "pred": r["relationship"], "correct": r["gold"] == r["relationship"],
                     "premise": r["premise"], "hypothesis": r["hypothesis"], "explanation": r["explanation"],
                     "premise_len": len(r["premise"].split()),
                     **{f"r{k + 1}": float(mat[k, col]) for k in range(3)}, "quality": q,
                     "plausible": q >= PLAUSIBLE, **m, **{f"fail_{k}": v for k, v in fails.items()},
                     "fails_any": any(fails.values()) if fails else None,
                     **{f"short_{s}": v for s, v in short.items()}})
    missing = sum(1 for r in key if (r["run"], r["id"]) not in fa)
    if missing:
        print(f"\n  WARNING: {missing} items have no faithfulness record")
    with open(os.path.join(a.out_dir, "ratings_merged.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n== 3. Quality per model (mean of 3 raters; plausible = mean >= 2/3)")
    report["quality"] = {}
    for mod in MODELS + ["all"]:
        rs = [r for r in rows if mod in ("all", r["model"])]
        qs = np.array([r["quality"] for r in rs])
        byl = {l: float(np.mean([r["quality"] for r in rs if r["pred"] == l]))
               for l in ("entailment", "neutral", "contradiction")}
        report["quality"][mod] = {"n": len(rs), "mean": float(qs.mean()), "plausible_rate": float((qs >= PLAUSIBLE).mean()),
                                  "by_pred_label": byl,
                                  "correct_pred": float(np.mean([r["quality"] for r in rs if r["correct"]])),
                                  "wrong_pred": float(np.mean([r["quality"] for r in rs if not r["correct"]]))
                                  if any(not r["correct"] for r in rs) else None}
        x = report["quality"][mod]
        print(f"  {mod:16s} n={len(rs):3d} mean {x['mean']:.3f} plausible {x['plausible_rate']:.0%}  "
              f"E {byl['entailment']:.2f} N {byl['neutral']:.2f} C {byl['contradiction']:.2f}  "
              f"right {x['correct_pred']:.2f} wrong {x['wrong_pred'] if x['wrong_pred'] is None else round(x['wrong_pred'], 2)}")

    # 4. RQ2
    print("\n== 4. RQ2: Spearman rho(quality, faithfulness measure)  [95% bootstrap CI]  n")
    report["rq2"] = {"pooled": {}, "per_model": {}}
    for meas in MEASURES:
        res = spearman([r[meas] for r in rows], [r["quality"] for r in rows])
        report["rq2"]["pooled"][meas] = res
        line = f"  {meas:30s} " + (f"{res['rho']:+.3f} [{res['ci'][0]:+.2f},{res['ci'][1]:+.2f}] p={res['p']:.3f} n={res['n']}"
                                   if "rho" in res else f"n={res['n']} (too few)")
        print(line)
    print("  per model (rho, n):")
    for mod in MODELS:
        rs = [r for r in rows if r["model"] == mod]
        report["rq2"]["per_model"][mod] = {}
        cells = []
        for meas in MEASURES:
            res = spearman([r[meas] for r in rs], [r["quality"] for r in rs])
            report["rq2"]["per_model"][mod][meas] = res
            cells.append(f"{meas[:14]}={res['rho']:+.2f}" if "rho" in res else f"{meas[:14]}=  -  ")
        print(f"    {mod:16s} " + "  ".join(cells))
    # model level (n=4): mean human quality vs full-test-set faithfulness from summary.json is in the report;
    # here, the rated sample's own means
    report["rq2"]["model_level"] = {}
    for mod in MODELS:
        rs = [r for r in rows if r["model"] == mod]
        report["rq2"]["model_level"][mod] = {
            "quality": float(np.mean([r["quality"] for r in rs])),
            **{meas: float(np.mean([float(r[meas]) for r in rs if r[meas] is not None]))
               for meas in MEASURES if any(r[meas] is not None for r in rs)}}

    # 5. RQ3
    print("\n== 5. RQ3: plausible (quality >= 2/3) but failing a faithfulness test")
    rq3 = {"definition": "quality >= 2/3 and fails_any (" + ", ".join(FAILS) + ")"}
    pl = [r for r in rows if r["plausible"]]
    pu = [r for r in pl if r["fails_any"]]
    rq3["n_items"], rq3["n_plausible"], rq3["n_plausible_unfaithful"] = len(rows), len(pl), len(pu)
    rq3["rate_of_all"], rq3["rate_of_plausible"] = len(pu) / len(rows), len(pu) / max(1, len(pl))
    print(f"  plausible {len(pl)}/{len(rows)}; of those, failing a test: {len(pu)} ({rq3['rate_of_plausible']:.0%});"
          f" = {rq3['rate_of_all']:.0%} of all rated items")
    rq3["by_test"] = {}
    for k in FAILS:
        fl = [r for r in rows if r[f"fail_{k}"]]
        rq3["by_test"][k] = {"fails": len(fl), "of_which_plausible": sum(r["plausible"] for r in fl),
                             "mean_quality_fail": float(np.mean([r["quality"] for r in fl])) if fl else None,
                             "mean_quality_pass": float(np.mean([r["quality"] for r in rows if r[f"fail_{k}"] is False]))}
        x = rq3["by_test"][k]
        print(f"    {k:26s} fails {x['fails']:3d}, plausible among them {x['of_which_plausible']:3d}   "
              f"quality fail {x['mean_quality_fail'] if x['mean_quality_fail'] is None else round(x['mean_quality_fail'], 2)}"
              f" vs pass {x['mean_quality_pass']:.2f}")
    for grp, keyf in (("model", lambda r: r["model"]), ("pred", lambda r: r["pred"]),
                      ("premise_len", lambda r: "short (<=12)" if r["premise_len"] <= 12 else "long (>12)")):
        rq3[f"by_{grp}"] = {}
        for g in sorted({keyf(r) for r in rows}):
            gp = [r for r in pl if keyf(r) == g]
            rq3[f"by_{grp}"][g] = {"plausible": len(gp), "plausible_unfaithful": sum(r["fails_any"] for r in gp),
                                   "rate": sum(r["fails_any"] for r in gp) / max(1, len(gp))}
        print(f"  by {grp}: " + "  ".join(f"{g} {v['plausible_unfaithful']}/{v['plausible']} ({v['rate']:.0%})"
                                         for g, v in rq3[f"by_{grp}"].items()))
    rq3["premise_len_mean"] = {"plausible_unfaithful": float(np.mean([r["premise_len"] for r in pu])) if pu else None,
                               "plausible_faithful": float(np.mean([r["premise_len"] for r in pl if not r["fails_any"]]))}
    rq3["shortcomings"] = {s: {"plausible_unfaithful": sum(r[f"short_{s}"] > 0 for r in pu),
                               "other_plausible": sum(r[f"short_{s}"] > 0 for r in pl if not r["fails_any"]),
                               "all": sum(r[f"short_{s}"] > 0 for r in rows)} for s in SHORTCOMINGS}
    print("  shortcomings ticked (any rater; only rater 3 ticked any):",
          "  ".join(f"{s}: PU {v['plausible_unfaithful']} / other-plausible {v['other_plausible']} / all {v['all']}"
                    for s, v in rq3["shortcomings"].items()))
    ex = sorted(pu, key=lambda r: -r["quality"])
    random.Random(0).shuffle(ex)
    rq3["examples"] = [{k: r[k] for k in ("item", "model", "gold", "pred", "premise", "hypothesis", "explanation",
                                          "quality")} | {"fails": [k for k in FAILS if r[f"fail_{k}"]]} for r in ex[:12]]
    print("  examples:")
    for e in rq3["examples"][:6]:
        print(f"    [{e['model']}, {e['pred']}, q={e['quality']:.2f}, fails {','.join(e['fails'])}]\n"
              f"      P: {e['premise']}\n      H: {e['hypothesis']}\n      E: {e['explanation']}")
    report["rq3"] = rq3

    with open(os.path.join(a.out_dir, "rating_analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {a.out_dir}/ratings_merged.csv and rating_analysis.json")


if __name__ == "__main__":
    main()
