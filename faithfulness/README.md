# Faithfulness-Aware Self-Rationalization

B.Tech Project (DTU CSE, 2026–27). Builds on the self-rationalization models of
Yang et al., TACL 2025 (the rest of this repository).

**Question:** when a model explains its NLI prediction, does the prediction
depend on the evidence the explanation cites? We edit that evidence and check
whether the prediction reacts, comparing against matched edits to other words.

## Files

| File | Role |
|---|---|
| `data.py` | load e-SNLI (original CSVs, with human-highlighted evidence) |
| `evidence.py` | which premise/hypothesis words an explanation cites |
| `interventions.py` | counterfactual variants + matched controls |
| `stance.py` | which label an explanation argues for (rules, validated on human explanations) |
| `predictors.py` | T5 self-rationalizer (same format as the original repo); fast label-only scoring |
| `metrics.py` | flip rates, confidence drops, paired contrasts, 95% bootstrap CIs |
| `build_dataset.py` | model-independent counterfactual dataset from human highlights |
| `run_experiment.py` | full experiment for one model; resumable |
| `plots.py` | figures (`figures/*.png`) and tables (`tables/*.csv`) for a run |
| `experiments_colab.ipynb` | runs everything on Colab |

## Pipeline

```
e-SNLI pair ─► model: label + explanation ─► cited words (premise and hypothesis)
           ─► variants, per side:
                remove_cited      vs  remove_random (other content words)
                                  vs  remove_overlap (other words shared with the other sentence)
                isolate_cited     vs  isolate_random                  (premise only)
                remove_gold       vs  remove_gold_random               (human highlights)
           ─► model: label on every variant ─► metrics + figures
```

## Metrics

| Metric | Definition | Faithful if |
|---|---|---|
| Necessity | flip(remove_cited) − flip(control), paired | > 0 (CI excludes 0) |
| Sufficiency | keep(isolate_cited) − keep(isolate_random), paired | > 0 |
| Confidence version | same with the drop in P(original label) | > 0 |
| Label–explanation consistency | stance(explanation) == predicted label | high |
| Evidence alignment | precision/recall/F1 of cited words vs human highlights | (plausibility, not faithfulness) |
| Baseline | accuracy, macro-F1, confusion matrix | |

All results are reported for all examples and by the model's **predicted** label.
Removing evidence is not expected to flip a *neutral* prediction, so necessity is
read mainly for entailment and contradiction.

## Dataset

`build_dataset.py` turns the 9,824 e-SNLI test pairs into **37,295 counterfactual
variants** using the human-highlighted evidence (removal, isolation, matched
random controls; premise and hypothesis). It does not depend on any model, so it
is the shared test set for every model we evaluate.

The stance rules agree with the gold label on **88.0%** of the human e-SNLI test
explanations (`*_stats.json`), which is the reference point for the model's
label–explanation consistency.

## Running

```bash
python faithfulness/build_dataset.py --split test
python faithfulness/run_experiment.py --model_path <ckpt_dir> --n 2000 --out_dir <run_dir>   # GPU
python faithfulness/plots.py --run_dir <run_dir>

python faithfulness/run_experiment.py --dry_run --n 300    # pipeline test, no model; numbers meaningless
```

On Colab: open `experiments_colab.ipynb` (branch `faithfulness-pilot`).

## Known limitations
- e-SNLI neutral pairs have **no premise highlights** (annotators highlighted only
  the hypothesis), so premise gold variants exist only for entailment/contradiction.
- Cited evidence is found by word overlap (lower-case, light stemming, stopwords
  removed); paraphrased evidence is missed.
- Deleting words can make a sentence ungrammatical; the matched controls measure
  how much of an effect is just "damaged input".
- `remove_overlap` exists only when non-cited words shared with the other sentence
  are available, so its n is smaller.

## Next steps
- [x] Pilot with T5-Large (300 examples)
- [x] Matched controls, both sides, consistency metric, bootstrap CIs, figures
- [ ] Full e-SNLI test run for T5-Large (nt8)
- [ ] More models (other T5 checkpoints, OLMo-7B, a zero-shot instruction LLM) → model comparison
- [ ] Contradiction intervention (antonym replacement / LLM rewrite)
- [ ] Counterfactual correctness with counterfactually-augmented SNLI (Kaushik et al., 2020)
- [ ] Acceptability score per model → correlation with faithfulness (RQ3)
- [ ] Manual check of ~50 flagged examples
