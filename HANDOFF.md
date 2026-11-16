# Handoff: GP115 faithfulness project (as of 16 Nov 2026)

> **Repo layout note (for Claude Code on the web):** this copy lives at the repo root. Paths below written as `btech project/...` refer to the original laptop folder. In this repo: the research plan and other PDFs are in `docs/`, the earlier results are in `btp_results/`, and the code is in `faithfulness/`. The full e-SNLI CSV is in `data/esnli_dataset.csv.gz` (gzipped because the raw file is over 100 MB; run `gunzip -k data/esnli_dataset.csv.gz`).

## 1. The project
- DTU B.Tech project GP115, "Faithfulness-Aware Self-Rationalization: Detecting When LLM Explanations Do Not Support Their Own Predictions". Team: Amey Parashar (the user), Aman Tyagi, Bhavya Goel. Guide: Mr. Rohit Beniwal.
- The question: when a self-rationalizing NLI model gives a label and an explanation, does the label depend on the evidence the explanation cites? We edit that evidence in the input and check whether the prediction reacts, compared with matched edits to other words.
- It builds on Yang et al., TACL 2025 ("Self-rationalization in the wild"), repo UKPLab/tacl2025-ood-eval-self-rationalization. We use the authors' released **T5-Large checkpoints trained on e-SNLI** and do no training ourselves.
- Research plan: `Detailed_Research_Plan_GP115_corrected.pdf`.
  - Plan Table 1 lists three interventions: removal, isolation, contradiction.
  - Plan Table 2 lists the metrics: accuracy/macro-F1, evidence sensitivity, counterfactual correctness, explanation-evidence alignment, human rating, Spearman correlation.
  - RQ1 asks whether predictions react to the interventions. RQ2 asks whether quality correlates with faithfulness. RQ3 asks how often explanations are plausible but unfaithful.
  - The user says to ignore the per-member task assignments and the dates. The only date that matters is the **BTP-I report on 31 Dec 2026**.
- Compute: Google Colab with a T4 and Google Drive. The local machine is a Mac with no CUDA.

## 2. Where things are
| What | Where |
|---|---|
| Working code (active) | `btech project/BTech-Project/`, local branch `contradiction-intervention` |
| Pushed to | **github.com/amey75/Btech-Project**, branch `contradiction-intervention` (latest commit `57dad4e`) |
| Original team repo | github.com/Aman14o5/BTech-Project, branch `faithfulness-pilot` (`a03cb42`). **amey75 has no push access**; Aman can merge the branch later. |
| Older local clone (not used) | `btech project/tacl2025-ood-eval-self-rationalization/` (origin amey75, branch `stage1-pipeline`) |
| Results from earlier runs (laptop copy) | `btech project/btp_results/`: summary.json, per_example.csv, tables and figures. **No caches** (originals.jsonl / variant_preds.jsonl / records.jsonl). |
| Findings page (published artifact) | https://claude.ai/artifact/Waz1vFSXtTSvMjMFU5L83v ("GP115 Faithfulness Findings"), last updated 12 Nov. Contradiction results are not in it yet. |
| Project log written by the team | `BTech-Project/faithfulness/PROJECT_LOG.md` |
| Colab notebook | `faithfulness/experiments_colab.ipynb`. Open it via https://colab.research.google.com/github/amey75/Btech-Project/blob/contradiction-intervention/faithfulness/experiments_colab.ipynb |
| Checkpoints | In the user's Google Drive at `/content/drive/MyDrive/{nt8, fastvotek_nt8, random_nt128, fastvotek_nt128}`. These are shortcuts to the authors' public Drive folder and each one has config.json + pytorch_model.bin. |
| Results on Drive | `/content/drive/MyDrive/btp_results/`. The user's Drive started empty, so all four full runs are computed fresh there. |

Git commits are made as `amey75` with the GitHub no-reply address `142935200+amey75@users.noreply.github.com`; there is no global git identity, so pass it with `-c`. Never use the user's personal email. Only commit or push when the user asks.

## 3. Pipeline (`faithfulness/`)
`run_experiment.py` handles one model per run. Each step caches to the run folder, so the run can resume after a disconnect.
1. Generate a label and explanation for each e-SNLI test pair (9,824 pairs) → `originals.jsonl`.
2. Find the **cited words** with `evidence.cited_indices`: content words of the premise or hypothesis that also appear in the explanation (lower-cased, light stemming, stopwords removed).
3. Build the edited inputs (`interventions.py`), for each side (premise and hypothesis):
   - `remove_cited` vs `remove_random` and `remove_overlap` (the overlap control removes non-cited words that also occur in the other sentence)
   - `isolate_cited` vs `isolate_random` (premise only)
   - `remove_gold`, `isolate_gold` and `remove_gold_random`, built from the e-SNLI human highlights (model-independent)
   - **new:** `contradict_cited`, `contradict_random` and `contradict_gold` (section 5)
4. Label-only scoring on every edited input → `variant_preds.jsonl`. This uses the first decoder step only and gives the same labels and probabilities as full generation.
5. **New:** the NLI oracle labels the original pairs and every contradiction edit. Its cache is `<parent of out_dir>/oracle_cache/<model>.jsonl`, keyed by the pair text and shared by all model runs.
6. `metrics.summarise` → `summary.json`, using paired bootstrap CIs and grouping by the **model's predicted label**. Also written: `records.jsonl`, `per_example.csv`, `contradiction_edits.csv` and `spotcheck_contradiction.csv` (150 edits; written once and never overwritten).
7. `plots.py --run_dir` → `figures/fig01..07` plus `tables/contrasts.csv` and `tables/variants.csv`.

Other files:
- `stance.py`: rule-based check of which label an explanation argues for. It agrees with the gold label on 88% of the human e-SNLI test explanations, and is weakest on neutral (76%).
- `build_dataset.py`: the model-independent gold edit set.
- `predictors.py`: T5Predictor, NLIOracle, and dummy stand-ins for `--dry_run`.
- `swaps.py`: word substitutions for the contradiction edits (section 5).

Model input/output format: input `explain nli hypothesis: {h} premise: {p}`, output `{label} explanation: {text}`. "nt8" means 8 examples **per label**, so 24 in total; nt128 means 384. The training sets are in `samples/esnli/{random,fastvotek}/sub_0/{8,128}/train_select.json`.

## 4. Results so far (four models, full e-SNLI test set; removal and isolation only)
Models: T5-Large trained on e-SNLI, {random, FastVote-k} × {nt8, nt128}. Run folders: `t5_nt8_test_full` (random nt8), `t5_esnli_fastvotek_nt8_full`, `t5_esnli_random_nt128_full`, `t5_esnli_fastvotek_nt128_full`.

| Measure | random nt8 | FV-k nt8 | random nt128 | FV-k nt128 |
|---|---|---|---|---|
| Accuracy | 85.6% | 86.8% | 84.9% | 87.0% |
| Explanation agrees with own label (human 88%) | 63.3% | 72.0% | 86.4% | 89.7% |
| Premise removal, cited minus random (flip rate) | +0.201 | +0.259 | +0.236 | +0.279 |
| Contradiction preds: cited premise removal | +0.075 | +0.165 | +0.343 | +0.270 |
| Contradiction preds: gold premise removal | +0.320 | +0.359 | +0.381 | +0.359 |
| Entailment: hypothesis cited vs overlap | +0.002 | −0.006 | −0.017 | +0.007 |

Findings:
1. All four models have about the same accuracy, but how often the explanation agrees with its own label varies by 27 points.
2. Cited words matter more than random words in every model.
3. With nt8, contradiction explanations name the wrong evidence. The model does use the gold evidence; the explanation cites other words. Example: pool vs ocean, where random nt8 cites "child".
4. Neutral explanations are the least consistent: only 33% for random nt8.
5. On the hypothesis side, entailment explanations do no better than other shared words (about 0 in all four models).
6. 8 → 128 examples helps much more than random → FastVote-k.

## 5. Evidence contradiction (new this session; code done, full runs in progress)
- **The edit:** replace ONE evidence word, so the sentence stays grammatical. The first possible operation is used:
  - `align`: the word clashes with a word in the other sentence, so copy that word in (premise "pool" → "ocean").
  - `swap_shared`: the word occurs in both sentences, so replace it with an incompatible word (man → woman).
  - `swap`: any other evidence word, replaced with an incompatible word.

  The control uses the same operation on a non-cited word when possible. The code fixes a/an and singular/plural after the swap.
- **Substitutes** (`swaps.py`):
  - Curated groups: gender, age, colours, numbers, posture, place/time, common adjectives, common animals.
  - WordNet noun siblings, restricted to concrete nouns: the word must be mainly a common noun, its main sense must be concrete, and the substitute must occur at least 3 times in the e-SNLI test split. People are handled only by the curated groups. Acts count only if they are sports, games or dances.
  - The `align` relation also accepts WordNet cousins (path distance ≤ 3, lowest common hypernym depth ≥ 3), which is what makes pool/ocean match.
  - Known noise: compound nouns ("lawn mower" → "campus mower"), phrasal verbs ("sleeps down"), and a few odd cousin pairs (air/foot, home/field, instrument/key).
- **The expected label comes from the oracle, not from rules.** The oracle is `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`. Only edits where the oracle is confident (p ≥ 0.9, `--oracle_min_conf`) are scored. Its accuracy on the original pairs is reported in `summary.json["oracle"]`.
- **Metrics** (`summary["counterfactual_correctness"]`):
  - Contradiction: `oracle_changes_label` (the edit should change the label), `follows_when_should_change`, `stays_when_should_stay`, `correct`, and `contradiction_follow_contrasts` (follows, cited minus random, unpaired bootstrap).
  - Two new paired flip contrasts in `CONTRASTS`: "premise/hypothesis contradiction vs random".
  - Removal: strict (moves to neutral) and lenient (moves away), computed for entailment and contradiction predictions.
  - Isolation: label kept.
- **Validation:**
  - Local dry runs pass.
  - Rerunning an existing run folder computes only the new edits.
  - Local oracle test: 89% accuracy on 200 original pairs.
  - About 33 of 40 gold edits read cleanly on manual reading.
  - The gold set has about 9,500 contradiction edits.
- **Colab pilot** (random nt8, 300 pairs, folder `btp_results/pilot_contradiction_nt8`):
  - Accuracy 0.853, matching the original pilot.
  - Oracle accuracy 0.877 (0.909 when confident).
  - Premise side: the model follows 84% of cited edits vs 53% of control edits, i.e. **+0.318 [+0.13, +0.49]**.
  - Hypothesis side: +0.015, CI includes 0. This is consistent with Finding 5: explanations cite almost every hypothesis word, so few control words are left.
  - Oracle says the label should change for 48% of cited edits vs 23% of control edits.
  - The pilot exposed one bug: old/elderly were treated as incompatible. Fixed in `57dad4e`.
- **Current state:** the user is running the 5b "Full runs" cell on Colab for all four models (estimated 20–30 min each, computed fresh). The old notebook cell "5. Run the experiment (--n 2000)" and the old cells 6–9 should be skipped.

## 6. Next steps
1. When the full runs finish, read the four `summary.json` files. Add the contradiction results (follows / stays / should change, cited vs control, per label) and counterfactual correctness for removal and isolation to the findings artifact (read it first, then republish to the same URL).
2. Hand spot-check: the team labels `pilot_contradiction_nt8/spotcheck_contradiction.csv` (150 rows: is the edit natural, which label is right). Then report how often the edit and the oracle are right.
3. Regenerate explanations on the edited inputs, for a subset of about 2,000 pairs × removal, isolation and contradiction × 4 models. This gives the plan's **explanation-evidence alignment** metric: the new explanation mentions the changed word and argues for the new label (via `stance()`). It is not implemented yet. Proposed option: `--explain_variants N` with its own cache file `variant_expl.jsonl`.
4. RQ2 (quality vs faithfulness). The repo's `human_eval/` folder has 1,080 explanations, each rated by 3 annotators on Yes / Weakly Yes / Weakly No / No. They cover SICK, VitaminC and XSum (not e-SNLI): T5 trained on e-FEVER, and OLMo on both datasets. Option A: run the faithfulness tests on those rated instances. Option B: rate about 50 of our e-SNLI explanations per model ourselves and compute Spearman's ρ against per-example faithfulness, e.g. the `_drop` columns.
5. Optional:
   - More T5 checkpoints (nt16/32/64, other selection strategies) to get enough models for a model-level correlation.
   - OLMo-7B. `load_custom_dataset.py` was reconstructed and its prompt is a guess; recover the real prompt from the authors' released OLMo outputs.
   - RQ3: logistic regression on `suspect_unfaithful`.
6. BTP-I report due 31 Dec 2026.

## 7. Gotchas
- Local testing used a venv with nltk, WordNet, torch and transformers in the session scratchpad, which will be gone. The Mac's Python 3.14 has an SSL certificate problem with `nltk.download`; download `wordnet.zip` with curl from `raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip` into `nltk_data/corpora/`, then `chmod -R go-w` the folder. Set `NLTK_DATA`. e-SNLI test CSV: `raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv`.
- Colab: every cell now `%cd`s or `os.chdir`s to `/content/BTech-Project`, because a session restart resets the folder to `/content`. Cell 1 no longer installs protobuf, since reinstalling it can trigger a restart.
- Removal and isolation keys and seeds are unchanged, so old caches stay valid. Contradiction edits use their own random seed (`"{seed}-{id}-{side}-contradict"`), and the substitution vocabulary always comes from the whole split, so edits do not depend on `--n`.
- The user prefers explanations in plain language, with real examples from the CSVs.
