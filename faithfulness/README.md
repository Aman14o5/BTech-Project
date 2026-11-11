# Faithfulness-Aware Self-Rationalization

B.Tech Project (DTU CSE, 2026–27). Builds on the self-rationalization models of
Yang et al., TACL 2025 (the rest of this repository).

**Question:** when a model explains its NLI prediction, does the prediction
depend on the evidence the explanation cites? We edit that evidence and check
whether the prediction reacts, comparing against edits to random words.

## Pipeline

```
e-SNLI test (CSV with human highlights)
  → model predicts label + explanation           predictors.py
  → words of the premise cited by the explanation evidence.py
  → counterfactual variants                      interventions.py
      remove_cited / isolate_cited               (the test)
      remove_random / isolate_random             (matched control)
      remove_gold / isolate_gold                 (human highlights, model-independent)
  → model predicts again on every variant
  → metrics                                      metrics.py
```

| Metric | Definition | Faithful if |
|---|---|---|
| evidence_sensitivity | flip rate(remove_cited) − flip rate(remove_random) | clearly > 0 |
| sensitivity_prob | prob. drop of original label, cited − random | clearly > 0 |
| sufficiency_gain | keep rate(isolate_cited) − keep rate(isolate_random) | clearly > 0 |
| cited vs gold P/R/F1 | overlap of cited words with human highlights | (alignment, not faithfulness) |

`per_example.csv` flags `suspect_unfaithful` examples: the cited words were
removed but the prediction and its probability barely moved.

## Running

```bash
# pipeline test without a model (numbers are meaningless)
python faithfulness/run_pilot.py --dry_run --n 50

# real run (GPU) with a fine-tuned T5 checkpoint from the authors' Drive
python faithfulness/run_pilot.py --model_path <checkpoint_dir> --n 300 --out_dir faithfulness/results/pilot_t5
```

On Colab, use `pilot_colab.ipynb`.

## Notes / known limitations
- In e-SNLI, neutral pairs have **no premise highlights** (annotators highlighted
  only the hypothesis), so `*_gold` variants exist only for entailment/contradiction.
- Cited evidence is found by word overlap (lower-case, light stemming,
  stopwords removed). Paraphrased evidence is missed.
- Deleting words can make the premise ungrammatical; the random control
  measures how much of the effect is just "damaged input".

## Next steps
- [ ] Pilot with T5-Large (e-SNLI, released checkpoint)
- [ ] Contradiction intervention (antonym replacement / LLM rewrite)
- [ ] Counterfactual correctness with counterfactually-augmented SNLI (Kaushik et al., 2020)
- [ ] More models: other T5 checkpoints, OLMo-7B, a zero-shot instruction LLM
- [ ] Acceptability score per model → correlation with faithfulness (RQ3)
