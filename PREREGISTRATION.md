# Pre-registration

Committed before any model was trained or evaluated. Inclusion values were fixed from `notebooks/01_eda.ipynb` using data distributions only (no model outputs). Anything not listed here is reported as *exploratory*. Deviations are logged at the bottom with a date and reason.

- Date: 2026-10-06
- Data: BirdCLEF+ 2025 `train_audio` / `train.csv` / `taxonomy.csv`; AnimalSpeak (`AnimalSpeak_limited.csv`, HF commit `4bc34f0`)

## Hypotheses

| | Hypothesis | Test |
|---|---|---|
| H1 | Per-species F1 of the supervised CNN (M2) increases with log(training recordings), and the effect remains after adjusting for recordings per recordist, mean duration, mean rating and iNaturalist share. | Spearman ρ; OLS slope of `F1 ~ log(n_rec) + n_rec/n_author + mean_duration + mean_rating + inat_share`; 95% bootstrap CI over species. Supported if the CI of the `log(n_rec)` slope excludes 0. |
| H2 | Per-species zero-shot recall of BioLingual (M3, prompt P1) increases with log(AnimalSpeak caption count), also on `unseen` recordings only. | Spearman ρ and slope with 95% bootstrap CI, on all recordings and on `unseen` recordings. |
| H3 | That slope is steeper for zero-shot (M3) than for a linear probe on the same embeddings (M4). | Bootstrap CI (over species) of slope(M3) − slope(M4) excludes 0. |
| H4 | Under-recorded species (lowest tertile of training recordings) need more survey recordings to be detected. | Detection probability 1 − (1 − recall)^k for k = 1…20, by tertile; descriptive. |

## Covariate structure (from EDA, before any model)

- `log(n_rec)` vs `log(n_author)`: Spearman ρ = 0.98 → the number of recordists cannot be adjusted for separately; it is replaced by recordings per recordist (ρ = 0.57 with n_rec).
- `log(n_rec)` vs `log(AnimalSpeak captions)`: ρ = 0.97 (sd of their log ratio 0.24). Downstream data volume and pretraining frequency are effectively **one exposure axis** in this dataset. Consequences:
  - H1 and H2 cannot be distinguished by their x-axis. H2 is interpreted only through M3, which receives no BirdCLEF training data, so any slope for M3 arises from pretraining or from species properties that co-vary with how often a species is recorded.
  - H3 (M3 vs M4 on the same embeddings and the same x) is therefore the main test of whether the language route adds disparity.
- Share of recordings already in AnimalSpeak (`seen`) is *higher* for under-recorded species (ρ = −0.31 with n_rec). If memorisation helps, it favours under-recorded species in the all-recordings analysis, i.e. it works *against* H2.

## Inclusion

- Taxon: `class_name == "Aves"`.
- Minimum recordings per species: **8**. Minimum distinct recordists: **3**. Reason: these were the defaults in the plan; the EDA grid is flat (any threshold in 5–10 recordings × 2–5 recordists keeps 145–146 of 146 bird species), so the defaults are kept. Result: 145 species, 27,642 recordings; 1 species (6 recordings) excluded.
- Recordings with missing `author` form their own group.
- Rating filter: **none**. All 6,608 iNaturalist recordings (and 588 Xeno-canto recordings) have rating 0 = unrated, so filtering would remove a whole source. `mean_rating` (covariate) is the mean over rated (> 0) recordings of each species.
- Excluded species and their recording counts are reported.

## Splits

- 5-fold `StratifiedGroupKFold`, group = recordist, stratified by `primary_label`, seed 0. Every fold's training part contains every included species.
- All metrics use out-of-fold predictions. M3 (no training) is evaluated on all included recordings.
- M2 epoch count is chosen once on fold 0 using a recordist-grouped validation split inside its training part, then fixed for all folds. Test folds are never used for model selection.

## Models

- M2: ImageNet EfficientNet-B0 fine-tuned on 128-bin log-mel of 5 s chunks (32 kHz), cross-entropy, time shift + SpecAugment, AMP, seed 0.
- M3: BioLingual (`davidrrobinson/BioLingual`), 48 kHz, same 5 s chunks, cosine similarity to prompt embeddings. Primary prompt P1 `"the sound of a {common name}"`; P2 `"the sound of {scientific name}"` for prompt-sensitivity analysis.
- M4: logistic regression on mean BioLingual audio embeddings, trained per fold.
- Chunks: up to 5 highest-energy 5 s chunks per recording, for training and evaluation. Recording-level prediction = mean of chunk probabilities.

## Metrics and statistics

- Recording-level top-1: per-species recall and F1, macro F1, accuracy; top-5 accuracy for M3.
- Species-level CIs: 1,000 bootstrap resamples of that species' recordings. Slope / correlation CIs: 1,000 bootstrap resamples of species.
- Leakage groups: `seen` (ID in AnimalSpeak), `unseen` (Xeno-canto ID > 798,785 or iNaturalist sound ID > 697,758, the maxima in AnimalSpeak), `unknown` (rest). H2 on `unseen` is restricted to species with ≥ 5 unseen recordings (128 of 145 species; 5,368 unseen recordings; median 7 per species in the lowest tertile, so those estimates are noisy). Primary H2 analysis on all recordings; confirmatory on `unseen`.
- Sensitivity: excluding recordings with secondary labels.
- Tertiles: by log(training recordings) among included species (low 11–89, mid 90–188, high 201–990 recordings).

## Not hypotheses (reported as exploratory)

Confusion analysis (where under-recorded species' errors go), class-weighted loss, noise robustness, IUCN overlap, Wikipedia / Xeno-canto exposure metrics.

## Deviations

| Date | Change | Reason |
|---|---|---|
