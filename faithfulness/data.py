"""Load e-SNLI (original CSV release) with human-highlighted evidence words.

The Hugging Face `esnli` dataset drops the highlights, so we read the CSVs from
https://github.com/OanaMariaCamburu/e-SNLI. `SentenceX_Highlighted_k` holds
0-based indices into `SentenceX.split()` ("{}" when nothing was highlighted).
Note: for neutral pairs annotators highlighted only the hypothesis, so the
premise highlight is always empty for neutral.
"""
import csv
import os
import random
import urllib.request

ESNLI_URL = "https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_{split}.csv"
LABELS = ["entailment", "neutral", "contradiction"]


def download_esnli(split="test", data_dir="faithfulness/data"):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"esnli_{split}.csv")
    if not os.path.exists(path):
        print(f"Downloading e-SNLI {split} to {path} ...")
        urllib.request.urlretrieve(ESNLI_URL.format(split=split), path)
    return path


def _parse_indices(field, n_tokens):
    field = (field or "").strip()
    if field in ("", "{}"):
        return []
    idx = sorted({int(i) for i in field.split(",") if i.strip().isdigit()})
    return [i for i in idx if i < n_tokens]


def load_esnli(path, n=None, seed=42, labels=LABELS):
    """Return a list of dicts: id, premise, hypothesis, label, explanation,
    gold_premise_idx, gold_hypothesis_idx (annotator 1)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["gold_label"] not in labels:
                continue
            premise = " ".join(r["Sentence1"].split())
            hypothesis = " ".join(r["Sentence2"].split())
            rows.append({
                "id": r["pairID"],
                "premise": premise,
                "hypothesis": hypothesis,
                "label": r["gold_label"],
                "explanation": r["Explanation_1"],
                "gold_premise_idx": _parse_indices(r["Sentence1_Highlighted_1"], len(premise.split())),
                "gold_hypothesis_idx": _parse_indices(r["Sentence2_Highlighted_1"], len(hypothesis.split())),
            })
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows
