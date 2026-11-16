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
| `interventions.py` | counterfactual variants + matched controls (removal, isolation, contradiction) |
| `swaps.py` | word substitutions for evidence contradiction (curated groups + WordNet) |
| `stance.py` | which label an explanation argues for (rules, validated on human explanations) |
| `predictors.py` | T5 self-rationalizer (same format as the original repo); fast label-only scoring; NLI oracle |
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
                contradict_cited  vs  contradict_random  (+ contradict_gold): one evidence word
                                      replaced by a conflicting one (man → woman, pool → ocean)
           ─► model: label on every variant
           ─► NLI oracle (DeBERTa-v3-large): expected label of every contradiction edit
           ─► metrics + figures
```

## Metrics

| Metric | Definition | Faithful if |
|---|---|---|
| Necessity | flip(remove_cited) − flip(control), paired | > 0 (CI excludes 0) |
| Sufficiency | keep(isolate_cited) − keep(isolate_random), paired | > 0 |
| Confidence version | same with the drop in P(original label) | > 0 |
| Contradiction necessity | flip(contradict_cited) − flip(contradict_random), paired | > 0 |
| Counterfactual correctness | removal: moves to neutral (strict) / moves away (lenient); isolation: label kept; contradiction: model label = oracle label, reported on edits that should change the label (*follows*) and that should not (*stays*) | high |
| Follows, cited vs control | follows(contradict_cited) − follows(contradict_random), unpaired | > 0 |
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
# evidence contradiction needs: pip install nltk && python -m nltk.downloader wordnet
# skip it with --no_contradiction
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
- Contradiction edits change one word. Substitutes come from curated groups (gender, age,
  colour, number, posture, place/time, common adjectives, common animals) and WordNet noun
  siblings/cousins (concrete nouns only; people only via the curated groups). Some edits are
  odd or do not really conflict ("hat" → "cap"). The rules do not decide the expected label: the
  oracle does, only confident oracle labels (p ≥ 0.9) are scored, the oracle's accuracy on the
  original pairs is reported in `summary.json`, and `spotcheck_contradiction.csv` (150 edits) is
  labelled by hand to measure how often edit and oracle are right.
- Contradiction edits are built only where a substitute exists, so their n is smaller than removal's.

## Evidence contradiction: how an edit is chosen
For the words in a set (cited, gold, or the non-cited control words) the first possible operation is used:
1. **align**: the word clashes with a word in the other sentence (premise *pool*, hypothesis *ocean*);
   copy that word in. Usually turns a contradiction into an entailment.
2. **swap_shared**: the word also occurs in the other sentence (*man* in both); replace it with an
   incompatible word (*woman*). Usually turns an entailment into a contradiction.
3. **swap**: any other evidence word, replaced with an incompatible word.

The control edits a non-cited word, using the same operation as the cited edit when possible.
Outputs per run: `contradiction_edits.csv` (every edit, oracle label, model label) and
`spotcheck_contradiction.csv` (written once and never overwritten, so it can be filled in on Drive).

## Next steps
- [x] Pilot with T5-Large (300 examples)
- [x] Matched controls, both sides, consistency metric, bootstrap CIs, figures
- [x] Full e-SNLI test run for T5-Large (4 checkpoints)
- [ ] More models (other T5 checkpoints, OLMo-7B, a zero-shot instruction LLM) → model comparison
- [x] Contradiction intervention (word substitution + NLI oracle), code; runs pending
- [ ] Manual spot-check of 150 contradiction edits (`spotcheck_contradiction.csv`)
- [ ] Regenerate explanations on edited inputs → explanation-evidence alignment
- [ ] Counterfactual correctness with counterfactually-augmented SNLI (Kaushik et al., 2020)
- [ ] Acceptability score per model → correlation with faithfulness (RQ3)
- [ ] Manual check of ~50 flagged examples
