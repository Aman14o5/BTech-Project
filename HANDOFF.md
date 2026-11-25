# Handoff: GP115 faithfulness project (as of 25 Nov 2026)

> **Where things are:** this file is at the repo root (github.com/amey75/Btech-Project, branches `main` and
> `contradiction-intervention`, which are identical) and is also copied to the laptop folder `btech project/`.
> In the repo: code in `faithfulness/`, results in `btp_results/`, PDFs in `docs/`, rating sheets in `human_rating/`,
> and the full e-SNLI CSV gzipped in `data/esnli_dataset.csv.gz` (run `gunzip -k` on it).

## 1. The project
- DTU B.Tech project GP115, "Faithfulness-Aware Self-Rationalization: Detecting When LLM Explanations Do Not
  Support Their Own Predictions". Team: Amey Parashar (the user), Aman Tyagi, Bhavya Goel. Guide: Mr. Rohit Beniwal.
- The question: when a self-rationalizing NLI model gives a label and an explanation, does the label depend on the
  evidence the explanation cites? We edit that evidence in the input and check whether the prediction (and the
  regenerated explanation) reacts, compared with matched edits to other words.
- It builds on Yang et al., TACL 2025 ("Self-rationalization in the wild"), repo UKPLab/tacl2025-ood-eval-self-rationalization.
  We use the authors' released **T5-Large checkpoints trained on e-SNLI** and do no training ourselves. (The plan says we
  would fine-tune; the report should say we used the released checkpoints.)
- Research plan: `docs/Detailed_Research_Plan_GP115_corrected.pdf`.
  - Table 1: interventions (removal, isolation, contradiction).
  - Table 2: metrics (accuracy/macro-F1, evidence sensitivity, counterfactual correctness,
    explanation-evidence alignment, human rating, Spearman).
  - RQ1: do predictions react to the interventions? RQ2: does quality correlate with faithfulness?
    RQ3: how often are explanations plausible but unfaithful, and what characterises those cases?
  - Ignore the plan's per-member tasks and dates. The only deadline is the **BTP-I report on 31 Dec 2026**.
    Aim for a full draft by about 24 Dec, so the guide can review it.
- Compute: Google Colab (T4) and Google Drive. The laptop is a Mac with no CUDA.
- **No more Colab runs are needed.** All model runs are finished.

## 2. Status at a glance
| Item | Status |
|---|---|
| Removal, isolation, contradiction on 4 models, full e-SNLI test set (9,824 pairs) | ✅ done, results committed |
| Accuracy, evidence sensitivity, counterfactual correctness (plan Table 2) | ✅ done |
| Explanation-evidence alignment (regenerated explanations, 2,000 pairs × 4 models) | ✅ done, results committed |
| Same-edit-type ("op-matched") contradiction contrasts | ✅ in every `summary.json` |
| Human rating sheets (200 items) | ✅ all three raters finished (200/200), committed |
| Contradiction spot-check (150 edits) | ❌ not done yet (one person, about 1 h); parked by the user on 25 Nov |
| RQ2 (Spearman) and RQ3 (plausible-but-unfaithful) analysis | ✅ first pass done (`faithfulness/analyse_ratings.py`, results in `btp_results/human_rating/`); see §6 |
| RQ2/RQ3 figure | ✅ `btp_results/human_rating/figures/fig08_rq2_rq3.{png,pdf}` (`faithfulness/plot_ratings.py`) |
| Contradiction figure for the report | ❌ to do (laptop only, from committed CSVs) |
| Findings artifact update | ❌ to do (https://claude.ai/artifact/Waz1vFSXtTSvMjMFU5L83v, last updated 12 Nov; read it before republishing to the same URL) |
| BTP-I report | ❌ not started; the main remaining work |

## 3. Where things are
| What | Where |
|---|---|
| Code (active) | laptop `btech project/BTech-Project/`, local branch `contradiction-intervention`; pushed to `amey/contradiction-intervention` and `amey/main` (latest commit `b3a1788`) |
| Original team repo | github.com/Aman14o5/BTech-Project, branch `faithfulness-pilot` (`a03cb42`). amey75 has no push access. |
| Old local clone (not used) | `btech project/tacl2025-ood-eval-self-rationalization/`, branch `stage1-pipeline` (already on GitHub) |
| Results (committed) | `btp_results/<run>/`: summary.json, per_example.csv, contradiction_edits.csv, spotcheck_contradiction.csv, variant_explanations.csv, originals.jsonl, figures/, tables/. Also mirrored to laptop `btech project/btp_results/`. |
| Large caches (Drive only) | `/content/drive/MyDrive/btp_results/<run>/`: records.jsonl (~60 MB), variant_preds.jsonl (~25 MB), variant_expl.jsonl, plus `oracle_cache/` |
| Run folders | `t5_nt8_test_full` (random nt8), `t5_esnli_fastvotek_nt8_full`, `t5_esnli_random_nt128_full`, `t5_esnli_fastvotek_nt128_full`, `pilot_contradiction_nt8` (300-pair pilot) |
| Checkpoints | Drive `/content/drive/MyDrive/{nt8, fastvotek_nt8, random_nt128, fastvotek_nt128}` (shortcuts to the authors' public folder) |
| Colab notebook | `faithfulness/experiments_colab.ipynb`; https://colab.research.google.com/github/amey75/Btech-Project/blob/main/faithfulness/experiments_colab.ipynb. Cells 1–2 are setup, cell 5b is the contradiction full runs and cell 5c is alignment. Old cells 5 and 6–9 only display things; skip them. |
| Project log written by the team | `faithfulness/PROJECT_LOG.md` (not updated this session) |

Git: commit as `amey75` with `-c user.name=amey75 -c user.email=142935200+amey75@users.noreply.github.com`
(there is no global identity). Never use the user's personal email. Push to remote `amey` (both branches:
`git push amey contradiction-intervention && git push amey contradiction-intervention:main`).
Only commit or push when the user asks. The repo is **public**; the user chose that knowingly.

## 4. Pipeline (`faithfulness/`)
`run_experiment.py` handles one model per run. Every step caches to the run folder, so a rerun computes only what is new.
1. Generate a label and explanation for each pair → `originals.jsonl`.
2. Cited words (`evidence.cited_indices`): content words of the premise or hypothesis that also appear in the explanation.
3. Edited inputs (`interventions.py`), per side:
   - removal: `remove_cited` vs `remove_random` and `remove_overlap`
   - isolation: `isolate_cited` vs `isolate_random` (premise only)
   - gold versions: `*_gold` from the e-SNLI human highlights
   - contradiction: `contradict_cited` / `_random` / `_gold`; one word is replaced via `align`, `swap_shared` or `swap` (`swaps.py`)
4. Label-only scoring of every edit → `variant_preds.jsonl`.
5. NLI oracle (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) gives the expected label of each
   contradiction edit. Only edits with p ≥ 0.9 are scored. Oracle accuracy on the original pairs: 0.890 (0.921 when confident).
6. **New: `--explain_variants N`** regenerates explanations on 8 edit types (`alignment.EXPLAIN_KEYS`) for a fixed
   subset of N pairs. `run_experiment.explain_subset(ids, 2000, 42)` gives the same pairs for every model.
   Output: `variant_expl.jsonl` (cache) and `variant_explanations.csv`.
7. `metrics.summarise` → `summary.json`, with bootstrap CIs, grouped by the model's predicted label. New keys:
   - `counterfactual_correctness.contradiction_follow_contrasts_op_matched`: same-edit-type comparison (`metrics.stratified_diff_ci`)
   - `explanation_alignment`: per variant, with subsets `all` / `label_changed` / `label_kept` / `should_change`,
     plus `should_change_by_op`, `contrasts` and `contrasts_op_matched`
8. `plots.py --run_dir` → figures fig01–07 and tables. There is **no contradiction or alignment figure yet**.
   `plot_ratings.py --dir btp_results/human_rating` → fig08 (A: RQ2 ρ per measure with CIs; B: RQ3 plausible-but-unfaithful per model).

**Alignment checks** (`alignment.py`), applied to each regenerated explanation:
- `mentions_new`: it names the substitute word.
- `cites_absent`: it names a word the edit took away.
- `consistent`: `stance()` agrees with the new label.
- `aligned` = consistent AND NOT cites_absent, plus mentions_new for contradiction edits.

Other files: `stance.py` (rule-based stance; 88% on human explanations, weakest on neutral), `build_dataset.py`,
`predictors.py` (T5Predictor, NLIOracle and dry-run dummies), `build_rating_sheets.py` (new).

Model I/O: input `explain nli hypothesis: {h} premise: {p}` → output `{label} explanation: {text}`.
"nt8" = 8 examples per label (24 in total); nt128 = 384.

## 5. Results (all four models, full test set)
Baseline and removal/isolation (unchanged from the earlier runs):

| Measure | random nt8 | FV-k nt8 | random nt128 | FV-k nt128 |
|---|---|---|---|---|
| Accuracy | 85.6% | 86.8% | 84.9% | 87.0% |
| Explanation agrees with own label (human 88%) | 63.3% | 72.0% | 86.4% | 89.7% |
| Premise removal, cited minus random (flip rate) | +0.201 | +0.259 | +0.236 | +0.279 |
| Removal → entailment moves away, cited/random | 70%/24% | 85%/24% | 59%/20% | 73%/17% |
| Isolation keeps contradiction label, cited/random/gold | 58/67/93% | 62/60/90% | 78/41/84% | 80/50/91% |
| Entailment: hypothesis cited vs overlap | +0.002 | −0.006 | −0.017 | +0.007 |

**Contradiction (RQ1):** the model follows an edit that should change the label.

| Premise contradiction | random nt8 | FV-k nt8 | random nt128 | FV-k nt128 |
|---|---|---|---|---|
| Follows, cited / control | 83% / 58% | 87% / 54% | 64% / 42% | 80% / 60% |
| Raw difference | +0.25 | +0.33 | +0.22 | +0.19 |
| **Same edit type (report this)** | +0.03 | +0.06 | +0.09 | +0.04 |
| Plain `swap` edits only | +0.20 | +0.28 | +0.14 | +0.29 |
| Gold edit, follows | 77% | 78% | 65% | 76% |

On the hypothesis side the difference is about 0 to +0.07 (same as Finding 5).

**The confound:** cited edits are about 85% `swap_shared` (followed 85–90% whatever word is edited); control edits
are about 45% plain `swap` (followed 21–30%). The raw gap is therefore mostly edit-type mix. The robust signal is
plain swaps on the premise. Always report the same-edit-type numbers.

**Explanation-evidence alignment** (regenerated explanations, about 1,950–1,980 usable pairs per model):

| | random nt8 | FV-k nt8 | random nt128 | FV-k nt128 |
|---|---|---|---|---|
| Premise contradiction aligned, cited / control | 47% / 9% | 49% / 13% | 64% / 22% | 67% / 16% |
| Same-edit-type difference | +0.30 | +0.11 (CI includes 0) | +0.16 | +0.24 |
| **Uncited premise word flipped the label → new explanation names it** | 46% | 44% | 71% | 68% |
| Cited edit, label kept → explanation still names the new word (copying) | 67% | 53% | 73% | 72% |
| Consistent after removal (remove_cited) | 49% | 52% | 75% | 81% |
| Explanation still cites a deleted word (remove_cited) | 3.5% | 5.3% | 4.2% | 2.0% |

**Caveats:**
- `mentions_new` is partly copying. Example: car→motorcycle, the label stays entailment, and the explanation becomes
  "a motorcycle is loaded with items on the top".
- The cleanest alignment number is the "uncited word flipped the label" row, taken from the `label_changed` subset.
- Word matching undercounts indirect references (outside→inside explained as "cannot be playing music outdoors"),
  so those numbers are lower bounds.

**Findings so far:**
1. All four models have about the same accuracy, but explanation-label agreement varies by 27 points.
2. Cited words matter more than random words for removal and isolation in every model.
3. With nt8, contradiction explanations cite the wrong evidence: isolating the cited words keeps the label no more
   often than random words, while the gold words keep it 90%+ of the time (example: pool vs ocean, where random nt8 cites "child").
4. Neutral explanations are the least consistent.
5. On the hypothesis side, entailment explanations do no better than other shared words.
6. Going from 8 to 128 examples helps much more than random → FastVote-k.
7. **New:** in contradiction, the raw cited-vs-control gap is mostly edit-type mix. A real but smaller effect remains
   on premise plain swaps.
8. **New:** when an uncited word changes the decision, nt8 explanations name it less than half the time and often
   invent another reason (children→adults flips the label to contradiction; the explanation says "the project cannot be a model").
   nt128 does better (about 70%). This is good RQ3 material.
9. **New:** random nt128 is the least reactive model (it follows 64% of edits that should change the label, vs 80–87% for the others).

## 6. Human rating (RQ2/RQ3)
- `human_rating/rating_sheet_rater{1,2,3}.xlsx`: the same 200 items, each in a different order, with dropdowns.
  50 per model, different pairs for each model, all from the 2,000-pair alignment subset, stratified by predicted
  label and `suspect_unfaithful` (25 suspect per model).
- Same task as Yang et al. (copied from `tacl2025-ood-eval-self-rationalization/human_eval/exp_eval2/`): "Does the
  explanation justify the given relationship?" Yes / Weakly Yes / Weakly No / No → 1, 2/3, 1/3, 0. Plus 5 optional
  shortcoming columns (H–L) and Notes.
- `human_rating/rating_key.csv` maps items to model, pair id, gold label and flags. Raters must not open it.
- `human_rating/README.md` holds the instructions for the team. The sheets were made by `faithfulness/build_rating_sheets.py` (needs openpyxl).
- **State on 25 Nov:** all three raters are complete (200/200, valid values) and committed.
  Rater 1 filled the sheet in Apple Numbers. Their ratings were copied unchanged into `rating_sheet_rater1.xlsx`
  (the `.numbers` copies of raters 1 and 2 stay uncommitted on the laptop; rater 2's is identical to their xlsx).
  Only rater 3 ticked shortcomings (79 rows); raters 1 and 2 ticked none, and nobody wrote notes.
- **Analysis:** `python faithfulness/analyse_ratings.py --out_dir btp_results/human_rating` (needs numpy, scipy,
  openpyxl; numbers-parser only to read a `.numbers` sheet). Writes `ratings_merged.csv` (one row per item: the 3
  scores, mean quality, every faithfulness measure, test failures, shortcomings), `rating_analysis.json` and `report.txt`.
- **Agreement:** Krippendorff's α = 0.45 ordinal / 0.45 interval (95% CI 0.36–0.53), moderate. Pairwise Spearman
  0.39–0.59; exact agreement 38–44%; within one step 70–86%. Rater means 0.56 / 0.56 / 0.51.
- **Quality (mean of 3 raters; "plausible" = mean ≥ 2/3):**

  | | random nt8 | FV-k nt8 | random nt128 | FV-k nt128 |
  |---|---|---|---|---|
  | Mean quality | 0.42 | 0.50 | 0.69 | 0.56 |
  | Plausible | 34% | 36% | 64% | 44% |

  The sample is 50% suspect by design. Reweighting to the 2,000-pair subset's suspect rate barely changes this
  (0.40 / 0.51 / 0.69 / 0.60). Humans prefer random nt128, even though FV-k nt128 has higher label agreement.
- **RQ2 (pooled Spearman ρ, quality vs measure):** explanation agrees with its label +0.45; alignment rate of the
  regenerated explanations +0.35; `suspect_unfaithful` −0.32 (all p < 0.001). The intervention-based measures are
  about 0: remove_cited drop −0.01, cited − random drop +0.13 (CI includes 0), isolate keeps the label +0.16 (p = 0.06),
  contradiction aligned +0.07. **Message: people reward coherent explanations, not ones the prediction actually
  depends on.** Per-model ρ are noisy (n = 50; removal tests only apply to the ~33 non-neutral items per model).
- **RQ3:** 89/200 plausible. 57 of them (64%) fail ≥1 test, 17 fail ≥2, and 48 fail only intervention tests
  (not the stance check). The tests used are stance disagrees, removal-insensitive (no flip and drop < 0.05),
  isolation loses the label, and contradiction misaligned. Report the ≥2 number too, because the ≥1 rate depends
  on how loose the tests are. Example (FV-k nt128, quality 1.0): P "A woman with a vest and red shirt…",
  H "…full sleeved jacket", E "vest and jacket are different garments"; deleting "vest" barely moves the prediction.
  The shortcoming analysis rests on rater 3 alone.
- A human + LLM-judge option was offered earlier and is no longer needed. Never present LLM ratings as human ones.
  For the spot-check, the user asked whether Claude could fill it. Options given: an LLM-only check (labelled as
  such), a hybrid (a human does 50 rows and Claude all 150, with agreement reported), or humans do all 150.
  Not decided yet.

## 7. Next steps
1. ~~RQ2/RQ3 analysis~~ (first pass done 25 Nov). Figure fig08 done too. Still open: choosing the final RQ3
   definition (≥1 vs ≥2 failed tests).
2. Parked by the user on 25 Nov. Someone fills in the spot-check: `btp_results/pilot_contradiction_nt8/spotcheck_contradiction.csv` (150 rows: is the
   edit natural, and the correct label). Report how often the edit is natural and how often humans agree with the oracle.
   Known oracle error example: child "sitting"→"standing" on a man's shoulders vs "standing by the man", which the oracle calls entailment.
3. Make the contradiction figure for the report: follows cited vs control per model, raw, same edit type and swap-only.
   Possibly an alignment figure too.
4. Update the findings artifact with the contradiction and alignment results (read it first, republish to the same URL).
5. **BTP-I report** (main job). The introduction, related work, method and RQ1 results can be written now.
   - Related work: add Atanasova et al. (ACL 2023, "Faithfulness Tests for Natural Language Explanations"), which is
     close to our contradiction + alignment test. Also consider Turpin et al. 2023, Lanham et al. 2023 and Parcalabescu & Frank 2024.
     These were cited from memory, so verify the details.
   - Soften the plan's claim "none of the reviewed works tests…" ([3] Wiegreffe et al. tests label–rationale association).
   - Say we used the released checkpoints, not our own fine-tuning.
6. Optional: OLMo-7B; more T5 checkpoints for a model-level correlation.

## 8. Gotchas
- Local testing: the Mac's Python 3.14 has no nltk or openpyxl.
  - Make a venv in the scratchpad (`pip install openpyxl numpy scipy numbers-parser`).
  - Without nltk, `swaps.py` falls back to the curated word groups (fine for dry runs).
  - e-SNLI test CSV: `curl` it from `raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv`
    into `faithfulness/data/` (gitignored). urllib/nltk downloads hit an SSL certificate problem.
  - Dry run: `python3 faithfulness/run_experiment.py --dry_run --n 300 --explain_variants 50 --out_dir <scratch>`.
- Colab: every cell `%cd`s to `/content/BTech-Project` (a restart resets the folder). Cell 1 pulls `contradiction-intervention`.
  Results go straight to Drive; a disconnect mid-run is fine, since rerunning resumes from the caches.
- Getting results off Colab: zip only the small files on Drive, then download (records.jsonl is about 60 MB per run and is not needed locally).
- Removal and isolation keys and seeds never changed, so old caches stay valid. Contradiction edits use their own seed;
  the substitution vocabulary always comes from the whole split.
- The user prefers plain-language explanations with real examples from the CSVs, and short answers.
