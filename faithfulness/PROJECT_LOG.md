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
