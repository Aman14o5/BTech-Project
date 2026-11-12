# Project log: Faithfulness-Aware Self-Rationalization

What we did, when, and why. Newest entries go at the bottom.
Times are local (IST). Commit hashes refer to branch `faithfulness-pilot` on github.com/Aman14o5/BTech-Project.

---

## 2026-11-11: Setup

- **Repo connected.** The local copy of the authors' code (UKPLab, Yang et al., TACL 2025) is linked to our GitHub repo.
  - `origin` = Aman14o5/BTech-Project
  - `upstream` = UKPLab original
  - Our work lives on branch `faithfulness-pilot`; `main` is the untouched original.
- **Codebase reviewed.**
  - The authors' T5 code works.
  - Their OLMo code imports a file that is missing from their repo (`load_custom_dataset.py`), so OLMo is postponed.
  - `code/Inference.py` contains the authors' hard-coded W&B API key. We never use or import it.
- **Decision: no training needed.** We use the authors' already fine-tuned T5-Large checkpoints from their Google Drive and build new test data around them.

## 2026-11-11 17:58: Pilot code (commit `c01a248`)

- Added the `faithfulness/` folder:
  - load e-SNLI
  - find the words an explanation cites
  - build edited (counterfactual) inputs
  - compute metrics
  - Colab notebook
- **Fixes:**
  - `eb2fea2` (18:08): fall back to the `t5-large` tokenizer when the checkpoint has none.
  - `d6110a6` (18:17): clear error message when the model path is wrong.
    - Cause: on Colab the path still had the `<path-to>` placeholder.
    - The checkpoint is at `/content/drive/MyDrive/nt8`, a shortcut to the authors' `nt8` model (8 training examples per label).

## 2026-11-11: Pilot run (300 examples, Colab)

- Sanity check passed. Tattoo example: contradiction, "a woman cannot get a tattoo."
- **First observations:**
  - Contradiction explanations follow a template ("X cannot <hypothesis>").
  - Some explanations argue for a different label than the one predicted.
  - Neutral predictions rarely change.
  - Entailment may just be word overlap.
- **Weaknesses found:**
  - only 300 examples
  - only random-word controls
  - premise side only
  - no confidence intervals

## 2026-11-11 21:52: Version 2 of the experiments (commits `26d26c6`, `478dd88`)

- **Model-independent dataset** (`build_dataset.py`). Human-highlighted evidence from e-SNLI is used to build edited versions: **9,824 test pairs → 37,295 variants**.
- **Stronger controls:**
  - "remove other overlapping words", in addition to random words
  - edits on both the premise and the hypothesis
  - versions that edit the human-highlighted evidence
- **Label–explanation consistency check** (`stance.py`). The rules agree with 88% of human explanations, after being improved from 77%.
- **Statistics:** 95% bootstrap confidence intervals, and results grouped by the model's predicted label.
- **Figures** (`plots.py`): heatmaps, bar charts, transitions, scatter, consistency, alignment.
- **Speed and robustness:**
  - Fast label-only scoring, checked on Colab to match full generation.
  - Runs are resumable: results are cached on Drive.
- Removed the old pilot script and notebook.

## 2026-11-11: Run on 2,000 test examples (T5-Large nt8)

- Accuracy **0.866**; consistency **0.640**.
- First version of the four findings. The entailment overlap result was based on only n=67, so it was not yet significant.

## 2026-11-11: Full run on all 9,824 test examples (T5-Large nt8)

- **Run:** `/content/drive/MyDrive/btp_results/t5_nt8_test_full`. Generation took about 9 minutes and the 91,499 edited inputs about 15 minutes.
- **Code version:** commit `478dd88`.
- **Baseline:** accuracy **0.856**, macro-F1 **0.855**. Predicted labels: entailment 3,509, neutral 2,947, contradiction 3,368.
- **Explanations cite input words:** premise 94.3%, hypothesis 95.6%.
- **Explanation argues for its own label:** **0.633** [0.624, 0.643]. The human reference is 0.880.
- **Key contrasts** (cited minus control, flip-rate difference, 95% CI):

  | Contrast | All | Entailment | Neutral | Contradiction |
  |---|---|---|---|---|
  | Premise removal vs random | +0.201 | +0.435 | +0.059 | +0.075 |
  | Premise removal vs overlap | −0.003 | **−0.101 [−0.171, −0.034]** | −0.042 | +0.155 |
  | Premise isolation vs random | +0.149 | +0.412 | +0.136 | **−0.125** |
  | Hypothesis removal vs random | +0.191 | +0.141 | +0.166 | +0.265 |
  | Hypothesis removal vs overlap | +0.175 | +0.002 | +0.279 | +0.465 |
  | Gold premise removal vs random | +0.283 | +0.301 | −0.049 | **+0.320** |
  | Gold hypothesis removal vs random | +0.315 | +0.054 | +0.347 | +0.531 |

- **Findings (all confirmed on the full set):**
  1. **Explanations are partly faithful.** Cited words matter more than random words (+0.20).
  2. **Entailment explanations reflect word overlap.** The cited words matter *less* than other shared words (significant).
  3. **Contradiction explanations miss the premise evidence the model uses:** gold +0.32 vs cited +0.075, and isolating the cited words gives −0.125. On the hypothesis side they are faithful.
  4. **About 1 in 3 explanations argues for a different label** than the one predicted.

## Open items (as of 2026-11-11 23:36)

- [ ] Figures for the full run (notebook cells 6–9 with `RUN_DIR = .../t5_nt8_test_full`)
- [ ] Second model: a larger-`nt` T5 checkpoint from the authors' Drive. Waiting for the list of folder names.
- [ ] Side-by-side comparison script. Proposed but not written; it needs approval.
- [ ] Manual check of about 50 flagged examples
- [ ] Viva slides (viva on 2026-11-17)

## 2026-11-11 23:50: Checked progress against the Detailed Research Plan (GP115)
- **Done:** removal and isolation (RQ1, partly), accuracy/F1, evidence sensitivity, check of cited evidence against human highlights, results per label.
- **Gaps:**
  - contradiction intervention
  - explanations are not regenerated on the edited inputs (the plan's "explanation-evidence alignment" needs them)
  - human quality ratings and Spearman correlation (RQ2)
  - plausible-but-unfaithful catalogue (RQ3)
  - second model (FastVote-k / different fine-tuning data)
  - manual spot-checks
- **Deviation to note:** the plan names FastVote-k selection as the default. We still need to check which selection strategy the nt8 checkpoint used.

## 2026-11-11 23:53: Checkpoint details confirmed
- The `nt8` model used so far comes from **random** sample selection, trained on e-SNLI with 8 examples per label.
- The run folder `t5_nt8_test_full` = T5-Large, e-SNLI, random selection, nt8.
- The plan's default is FastVote-k. Next runs: FastVote-k nt8, then FastVote-k at the largest available size (up to nt128), plus an e-FEVER-trained checkpoint if one exists.

## 2026-11-12 20:42: Model plan for the viva
- **Design:** 2 strategies × 2 sizes = T5 · e-SNLI · {random, FastVote-k} × {nt8, nt128}. We use nt128 rather than nt512 because the research plan says "up to 128 examples".
- **Next run:** random nt128 → `btp_results/t5_esnli_random_nt128_full`.

## 2026-11-12 21:15: Full run, T5 · e-SNLI · random · nt128 (all 9,824 test pairs)
- **Run:** `btp_results/t5_esnli_random_nt128_full`. Code version `cbedb53` (experiment code unchanged since `478dd88`). 94,680 edited inputs.
- **Baseline:** accuracy **0.849**, macro-F1 **0.849**. Predicted labels: entailment 3,617, neutral 3,334, contradiction 2,873.
- **Explanations cite input words:** premise 97.3%, hypothesis 97.3%.
- **Consistency:** **0.864** [0.857, 0.871]. The human reference is 0.880.
- **Random nt8 → random nt128** (flip-rate difference, cited minus control):

  | Contrast | Label | nt8 | nt128 |
  |---|---|---|---|
  | Premise removal vs random | contradiction | +0.075 | **+0.343** |
  | Premise isolation vs random | contradiction | −0.125 | **+0.370** |
  | Premise removal vs overlap | entailment | −0.101 | **−0.058** (still significant) |
  | Hypothesis removal vs random | entailment | +0.141 | +0.014 (≈ 0) |
  | Gold premise removal vs random | contradiction | +0.320 | +0.381 |

- **Takeaways:**
  1. More training examples leave accuracy unchanged (0.856 → 0.849), but faithfulness and consistency improve a lot (0.633 → 0.864).
  2. The contradiction problem largely goes away at nt128: the explanation now names almost the same evidence the model uses (cited +0.34 vs gold +0.38).
  3. The entailment word-overlap problem remains at both sizes: a robust finding.

## 2026-11-12 21:47: Full run, T5 · e-SNLI · FastVote-k · nt8 (all 9,824 test pairs)
- **Run:** `btp_results/t5_esnli_fastvotek_nt8_full`. Code version `cbedb53`. 91,668 edited inputs.
- **Baseline:** accuracy **0.868**, macro-F1 **0.868**. Predicted labels: entailment 3,382, neutral 3,238, contradiction 3,204.
- **Explanations cite input words:** premise 92.5%, hypothesis 97.6%.
- **Consistency:** **0.720** [0.712, 0.728].
- **Three models side by side:**

  | Measure | random nt8 | FastVote-k nt8 | random nt128 |
  |---|---|---|---|
  | Accuracy | 0.856 | 0.868 | 0.849 |
  | Consistency | 0.633 | 0.720 | 0.864 |
  | Contradiction, premise removal vs random | +0.075 | +0.165 | +0.343 |
  | Contradiction, premise isolation vs random | −0.125 | +0.004 | +0.370 |
  | Contradiction, gold premise removal vs random | +0.320 | +0.359 | +0.381 |
  | Entailment, premise removal vs overlap | −0.101 | +0.000 | −0.058 |
  | Entailment, hypothesis removal vs overlap | +0.002 | −0.006 | −0.017 |

- **Takeaways:**
  1. At the same size, FastVote-k gives somewhat more faithful and more consistent explanations than random selection.
  2. More examples (nt128) helps more than the selection strategy.
  3. Accuracy barely differs across models (0.849–0.868), while faithfulness and consistency differ a lot.
  4. Entailment explanations are no better than overlap words in any model.

## 2026-11-12 22:20: Full run, T5 · e-SNLI · FastVote-k · nt128 (all 9,824 test pairs). The 2×2 is complete.
- **Run:** `btp_results/t5_esnli_fastvotek_nt128_full`. Code version `cbedb53`. 95,287 edited inputs.
- **Baseline:** accuracy **0.870**, macro-F1 **0.870**. Predicted labels: entailment 3,414, neutral 3,159, contradiction 3,251.
- **Explanations cite input words:** premise 96.4%, hypothesis 97.1%.
- **Consistency:** **0.897** [0.892, 0.903].
- **The four models:**

  | Measure | random nt8 | FastVote-k nt8 | random nt128 | FastVote-k nt128 |
  |---|---|---|---|---|
  | Accuracy | 0.856 | 0.868 | 0.849 | 0.870 |
  | Consistency | 0.633 | 0.720 | 0.864 | 0.897 |
  | Premise removal vs random (all) | +0.201 | +0.259 | +0.236 | +0.279 |
  | Contradiction, premise removal vs random | +0.075 | +0.165 | +0.343 | +0.270 |
  | Contradiction, premise isolation vs random | −0.125 | +0.004 | +0.370 | +0.295 |
  | Contradiction, gold premise removal vs random | +0.320 | +0.359 | +0.381 | +0.359 |
  | Entailment, premise removal vs overlap | −0.101 | +0.000 | −0.058 | **+0.092** |
  | Entailment, hypothesis removal vs overlap | +0.002 | −0.006 | −0.017 | +0.007 |

- **Takeaways:**
  1. Accuracy varies by only 2 points across the four models, while consistency varies by 26 points.
  2. More examples matter more than the selection strategy.
  3. FastVote-k nt128 is the best overall, and the only model whose entailment premise evidence beats overlap words.
  4. On the hypothesis side, entailment explanations never beat overlap words in any model.
  5. The contradiction gap closes with nt128 under both strategies.
