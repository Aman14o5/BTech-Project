"""Build the human-rating sheets for explanation quality (plan, Table 2: "Human rating"; RQ2, RQ3).

Same task as Yang et al. (TACL 2025, human_eval/exp_eval2 in the authors' repo): a rater sees the
premise, the hypothesis, the label the MODEL predicted ("relationship") and the model's explanation,
and answers "Given the premise and hypothesis, does the explanation justify the given relationship?"
on Yes / Weakly Yes / Weakly No / No (scores 1, 2/3, 1/3, 0), plus optional shortcomings.

Sample: --per_model explanations from each model, drawn from the fixed 2,000-pair subset used for
explanation regeneration (run_experiment.explain_subset), so every rated explanation also has the
alignment results. Pairs are different for every model (a rater never sees the same pair twice),
and stratified by predicted label and by the per_example.csv `suspect_unfaithful` flag, so both
faithful-looking and suspect explanations are rated. Raters are blind to the model, the gold label
and all faithfulness results; each rater gets the same items in a different order.

Writes to --out_dir:
  rating_sheet_rater{1,2,3}.xlsx   one per rater (dropdowns; opens in Excel or Google Sheets)
  rating_key.csv                   item -> model, pair id, gold label ... (raters must NOT open it)

Run from the repo root (needs openpyxl):
  python faithfulness/build_rating_sheets.py --results btp_results --out_dir human_rating
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import download_esnli, load_esnli
from run_experiment import explain_subset

LABELS = ["entailment", "neutral", "contradiction"]
MODELS = {  # name in the key -> run folder under --results
    "random_nt8": "t5_nt8_test_full",
    "fastvotek_nt8": "t5_esnli_fastvotek_nt8_full",
    "random_nt128": "t5_esnli_random_nt128_full",
    "fastvotek_nt128": "t5_esnli_fastvotek_nt128_full",
}
RATINGS = ["Yes", "Weakly Yes", "Weakly No", "No"]
SHORTCOMINGS = [  # the authors' options; "None" = leave all of them empty
    ("Does not make sense", "the explanation is nonsensical"),
    ("Insufficient", "insufficient justification: important reasons are missing"),
    ("Irrelevant", "irrelevant to the task"),
    ("Only repeats", "too trivial: only repeats one of the sentences"),
    ("Hallucinated", "contains content that is not in the premise"),
]

INSTRUCTIONS = [
    ("How to rate explanations", "title"),
    ("You will see a Premise, a Hypothesis, their Relationship, and an Explanation that an AI generated to "
     "justify the Relationship. Your task is to judge the quality of the Explanation.", ""),
    ("The three relationships:", "bold"),
    ("•  Entailment: the premise gives enough evidence to conclude that the hypothesis is TRUE.", ""),
    ("•  Contradiction: the premise gives enough evidence to conclude that the hypothesis is FALSE.", ""),
    ("•  Neutral: the premise does not give enough evidence to decide either way.", ""),
    ("Question 1 (required, column G): Given the premise and hypothesis, does the explanation justify the given "
     "relationship?  Pick Yes / Weakly Yes / Weakly No / No.", "bold"),
    ("An explanation justifies the relationship if it is easy to understand, gives the important reasons, and does "
     "NOT just repeat or restate one of the sentences.", ""),
    ("Rate the explanation for the relationship SHOWN, even if you think the relationship itself is wrong. "
     "(A good explanation of a wrong relationship can still justify it; say so in Notes if you like.)", ""),
    ("Question 2 (optional, columns H-L): what are its shortcomings? Put an x in every column that applies; "
     "leave them all empty if there are none.", "bold"),
] + [(f"•  {short}: {long}", "") for short, long in SHORTCOMINGS] + [
    ("Tips", "bold"),
    ("•  Ignore small grammar and style mistakes (capital letters, missing full stops, a missing word).", ""),
    ("•  Before reading the explanation, think of the explanation you would give a friend, and compare.", ""),
    ("•  Judge only how good the explanation is as a justification. Do not try to guess whether the AI 'really' "
     "used it; that is measured separately.", ""),
    ("•  Work on your own and do not discuss items with the others until all three of you have finished.", ""),
    ("•  Do not open rating_key.csv in the repo: it tells you which model wrote each explanation.", ""),
    ("•  Do not reorder, delete or add rows, and do not change the Item column. You can stop and continue later.", ""),
]


def read_per_example(path):
    with open(path, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def sample_model(rows, candidates, k, rng):
    """k ids from `candidates`, stratified by predicted label x suspect flag, as even as possible."""
    strata = {}
    for i in candidates:
        r = rows[i]
        if r["explanation"].strip():
            strata.setdefault((r["pred"], r["suspect_unfaithful"] == "True"), []).append(i)
    for v in strata.values():
        rng.shuffle(v)
    keys = [(l, s) for l in LABELS for s in (True, False)]
    chosen = []
    while len(chosen) < k and any(strata.get(key) for key in keys):
        for key in keys:  # round robin over the strata
            if len(chosen) < k and strata.get(key):
                chosen.append(strata[key].pop())
    return chosen


def write_sheet(path, items, order, rater):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    ins.column_dimensions["A"].width = 130
    for row, (text, style) in enumerate(INSTRUCTIONS, start=1):
        c = ins.cell(row=row, column=1, value=text)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if style == "title":
            c.font = Font(bold=True, size=14)
        elif style == "bold":
            c.font = Font(bold=True)
    ins.cell(row=len(INSTRUCTIONS) + 2, column=1, value=f"This sheet: rater {rater}, {len(order)} items. "
             "Rate in the 'Ratings' tab.").font = Font(italic=True)

    ws = wb.create_sheet("Ratings")
    header = (["#", "Item", "Premise", "Hypothesis", "Relationship", "Explanation", "Q1: Justifies? (required)"]
              + [s for s, _ in SHORTCOMINGS] + ["Notes"])
    widths = [5, 7, 45, 35, 14, 45, 16] + [12] * len(SHORTCOMINGS) + [30]
    fill = PatternFill("solid", fgColor="DDEBF7")
    for col, (h, w) in enumerate(zip(header, widths), start=1):
        c = ws.cell(row=1, column=col, value=h)
        c.font = Font(bold=True)
        c.fill = fill
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.column_dimensions[c.column_letter].width = w
    for n, idx in enumerate(order, start=1):
        it = items[idx]
        vals = [n, it["item"], it["premise"], it["hypothesis"], it["relationship"], it["explanation"]]
        for col, v in enumerate(vals, start=1):
            ws.cell(row=n + 1, column=col, value=v).alignment = Alignment(wrap_text=True, vertical="top")
    last = len(order) + 1
    rating_dv = DataValidation(type="list", formula1='"' + ",".join(RATINGS) + '"', allow_blank=True,
                               showErrorMessage=True, error="Pick Yes, Weakly Yes, Weakly No or No")
    x_dv = DataValidation(type="list", formula1='"x"', allow_blank=True, showErrorMessage=True,
                          error="Type x or leave empty")
    ws.add_data_validation(rating_dv)
    ws.add_data_validation(x_dv)
    rating_dv.add(f"G2:G{last}")
    first_x, last_x = ws.cell(row=1, column=8).column_letter, ws.cell(row=1, column=7 + len(SHORTCOMINGS)).column_letter
    x_dv.add(f"{first_x}2:{last_x}{last}")
    ws.freeze_panes = "C2"
    wb.active = 1
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="btp_results", help="folder holding the four run folders")
    ap.add_argument("--out_dir", default="human_rating")
    ap.add_argument("--per_model", type=int, default=50)
    ap.add_argument("--raters", type=int, default=3)
    ap.add_argument("--explain_n", type=int, default=2000, help="must match --explain_variants of the runs")
    ap.add_argument("--seed", type=int, default=42, help="must match --seed of the runs")
    ap.add_argument("--data_dir", default="faithfulness/data")
    args = ap.parse_args()

    ids = [e["id"] for e in load_esnli(download_esnli("test", args.data_dir))]
    subset = sorted(explain_subset(ids, args.explain_n, args.seed))
    rng = random.Random(f"{args.seed}-rating")
    rng.shuffle(subset)

    items, used = [], set()
    for model, folder in MODELS.items():
        rows = read_per_example(os.path.join(args.results, folder, "per_example.csv"))
        chosen = sample_model(rows, [i for i in subset if i not in used and i in rows], args.per_model, rng)
        used.update(chosen)
        for i in chosen:
            r = rows[i]
            items.append({"model": model, "run": folder, "id": i, "gold": r["gold"], "relationship": r["pred"],
                          "premise": r["premise"], "hypothesis": r["hypothesis"], "explanation": r["explanation"],
                          "suspect_unfaithful": r["suspect_unfaithful"], "consistent": r["consistent"]})
    rng.shuffle(items)  # item numbers must not reveal the model
    for n, it in enumerate(items, start=1):
        it["item"] = f"Q{n:03d}"

    os.makedirs(args.out_dir, exist_ok=True)
    for rater in range(1, args.raters + 1):
        order = list(range(len(items)))
        random.Random(f"{args.seed}-rater{rater}").shuffle(order)
        write_sheet(os.path.join(args.out_dir, f"rating_sheet_rater{rater}.xlsx"), items, order, rater)
    keep = ["item", "model", "run", "id", "gold", "relationship", "premise", "hypothesis", "explanation",
            "suspect_unfaithful", "consistent"]
    with open(os.path.join(args.out_dir, "rating_key.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)

    print(f"{len(items)} items -> {args.out_dir}/rating_sheet_rater1..{args.raters}.xlsx + rating_key.csv")
    for model in MODELS:
        its = [it for it in items if it["model"] == model]
        by = {l: sum(it["relationship"] == l for it in its) for l in LABELS}
        print(f"  {model:16s} n={len(its)}  {by}  suspect={sum(it['suspect_unfaithful'] == 'True' for it in its)}")


if __name__ == "__main__":
    main()
