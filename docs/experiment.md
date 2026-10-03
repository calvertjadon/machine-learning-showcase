# Experiment protocol and measured results

This document is the deep protocol record for the NFL play-type experiment
that produces `results/metrics.json`. Every reported measurement and provenance
fact below was checked against that file when this document was written (its
`metadata` block is the training manifest; `train` also writes a copy to the
local `artifacts/training.json`); the figures under `results/` are generated
from the same payload by `nfl-showcase evaluate`. The feature audit that the
models rely on is [features.md](features.md); the data source and cleaning
diagnosis is [data.md](data.md).

- **Task**: classify the type of a football play — one of `field_goal`,
  `pass`, `punt`, `qb_kneel`, `qb_spike`, `run` — from pre-play context only.
- **Data**: public *NFL Play by Play 2009-2018 (v5)* CSV (not distributed
  here; see [data.md](data.md)).
- **Split**: game-separated chronological holdout starting 2017-09-01.
- **Models**: a majority baseline and a down/distance baseline, logistic
  regression, and a random forest, all fitted on the training seasons only.
- **Status**: a single-holdout research measurement, not a production system
  and not a benchmark claim about football prediction in general.

## Reproduce the recorded run

After `uv sync --locked` and obtaining the file from `docs/data.md`:

```bash
uv run nfl-showcase train \
  --data "data/NFL Play by Play 2009-2018 (v5).csv" \
  --exclude-ambiguous-games \
  --output-dir artifacts \
  --holdout-start 2017-09-01 \
  --seed 42 --trees 200 --jobs 2

uv run nfl-showcase evaluate \
  --data "data/NFL Play by Play 2009-2018 (v5).csv" \
  --model artifacts/models.joblib \
  --output-dir results
```

`--exclude-ambiguous-games` is required for this source: without it, the
explicit cleaning step rejects the source's 22 conflicting identifiers (see
below). Both commands hash the CSV and `evaluate` refuses a file that does not
match the hash recorded by `train`, so the numbers cannot be silently
reproduced on a different release.

## Source, cleaning, and label support

The recorded run's source facts (the same values appear in the
`results/metrics.json` metadata; `train` writes the manifest locally to
`artifacts/training.json`):

| Fact | Value |
| --- | --- |
| Source file | `NFL Play by Play 2009-2018 (v5).csv` |
| Bytes / SHA-256 | 700,397,316 / `33708680002e8e57eb42b1f41ff45b214fac8422952a89bb27acbeb18726dc90` |
| Exact-duplicate rows removed | 2,393 (repeated across all 23 selected columns; first occurrence kept) |
| Ambiguity | 22 `(game_id, play_id)` groups disagreeing after deduplication, all in one game |
| Whole-game exclusion (opt-in) | 1 game, 268 rows dropped (`--exclude-ambiguous-games`) |
| Retained raw rows | 449,371 raw → 446,710 retained |

The ambiguous game is dropped whole, not row by row, so previous-play context
and whole-game evaluation never stitch across deleted plays. The default
`exclude_ambiguous_games=False` rejects the source instead of guessing; the
cleaning summary is recorded in the run metadata and replayed by `evaluate`.

Prepared rows keep a target row even when its numeric context is missing;
imputation happens inside the model pipelines, fitted on training rows only.
Rows whose `play_type` is not one of the six target labels (kickoffs, extra
points, `no_play`, missing) are filtered after previous-play context has been
derived from every original row, so context is never lost.

### Class support (prepared rows)

| Label | Train rows | Holdout rows |
| --- | ---: | ---: |
| `field_goal` | 7,860 | 1,861 |
| `pass` | 150,373 | 35,187 |
| `punt` | 19,447 | 4,359 |
| `qb_kneel` | 3,054 | 757 |
| `qb_spike` | 562 | 121 |
| `run` | 107,234 | 24,680 |
| **Total** | **288,530** | **66,965** |

The holdout is strongly imbalanced: `pass` is 52.5% of it, while `qb_spike` is
0.18% (121 rows). This is why the primary metrics are per-class and macro-F1
alongside accuracy.

## Split

`chronological_split(prepared, holdout_start="2017-09-01")`, one date per game
and every game kept whole:

| Partition | Rows | Games | Dates |
| --- | ---: | ---: | --- |
| Train | 288,530 | 2,045 | 2009-09-10 → 2017-01-01 |
| Holdout | 66,965 | 480 | 2017-09-07 → 2018-12-17 |

No `game_id` appears in both partitions, and no game is split across the
cutoff. A random row-level split would put plays from the same game and season
on both sides; this split measures generalization to later seasons instead.

## Pre-play feature audit (summary)

Each model receives the same 16 `FEATURE_COLUMNS`, all known before the snap
or derived from the previous play of the same game; complete rejection list and
rationale are in `docs/features.md`.

| Feature | Timing | Notes |
| --- | --- | --- |
| `yardline_100` | pre-snap | Distance to the opponent's goal line. |
| `quarter_seconds_remaining`, `half_seconds_remaining`, `qtr` | pre-snap | Clock and period. |
| `down`, `ydstogo` | pre-snap | Down and distance; missing values preserved (e.g. two-point tries). |
| `posteam_timeouts_remaining`, `defteam_timeouts_remaining` | pre-snap | Timeouts left; missing values preserved. |
| `score_differential` | pre-snap | Possession team minus opponent at the snap. |
| `previous_yards_gained` | previous play | Same-game shift of `yards_gained`; `NaN` on a game's first row or after a missing value. |
| `clock_running_proxy` | previous play | Conservative 0/1 estimate that the clock kept running after the previous play (never a reconstruction of the actual clock). |
| `home_team`, `away_team`, `posteam`, `defteam` | pre-snap | Normalized team abbreviations. |
| `previous_play_type` | previous play | Same-game shift of `play_type`, including special teams and `unknown`. |

Everything recorded after the snap — current-play yards, EPA/WPA, win
probability, outcome flags, post-play score, player identities, and current
`desc` (the current row's free text) — is excluded; `desc` is read only to
evaluate the *previous* row for the clock proxy. Previous-play values are
within-game `shift(1)` after a global `(game_id, play_id)` sort, so context
never crosses a game boundary and never reaches forward in time.

## Models and fitting protocol

All four estimators consume raw `FEATURE_COLUMNS` frames and are fitted on the
training frame only. The two simple baselines — `majority` and
`down_distance` — learn training-set label counts directly and apply no
preprocessing. The two learned pipelines wrap their preprocessing —
imputation, scaling, one-hot encoding — inside the estimator, so the
transforms are fitted on training rows only:

- **`majority`** — always predicts the training set's most frequent label
  (a class-prior baseline).
- **`down_distance`** — modal play type per `(down, yards-to-go bucket)` with
  buckets `<=2`, `<=5`, `<=10`, `>10`, falling back to the training prior;
  learned from training rows only, no outcome-based rules.
- **`logistic_regression`** — median numeric imputation + standard scaling,
  most-frequent categorical imputation + one-hot encoding
  (`handle_unknown="ignore"`), `max_iter=1000`.
- **`random_forest`** — same preprocessing, 200 trees, `max_depth=18`,
  `min_samples_leaf=5`, `random_state=42`, `n_jobs=2`.

The forest was the pre-selected default predictor before evaluation; the
holdout was not used to tune, select, or calibrate anything. Reported metrics
are a single measurement on the 2017-2018 games.

## Measured results

Holdout = 66,965 rows from 480 games. Values below are rounded to four
decimals for reading; `results/metrics.json` carries the full precision and
the per-class reports, confusion matrices, and supports that the figures use.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| `majority` | 0.5255 | 0.1148 |
| `down_distance` | 0.6069 | 0.3297 |
| `logistic_regression` | 0.6741 | **0.6734** |
| `random_forest` | **0.7093** | 0.6663 |

The forest has the highest accuracy; logistic regression has the higher
macro-F1. Neither dominates every class, and there is no universal best model
here — the choice depends on whether overall correctness or balanced
per-class performance matters.

Why macro-F1 separates them — the rare-class evidence:

- `majority` scores 0.5255 accuracy by predicting `pass` for every play; its
  macro-F1 is 0.1148 because the other five classes score zero F1.
- `down_distance` reaches 0.6069 accuracy and 0.3297 macro-F1 by learning the
  common `pass`/`run`/`punt` patterns; `field_goal`, `qb_kneel`, and
  `qb_spike` still score zero F1.
- `logistic_regression` recovers 34 of the 121 holdout `qb_spike` rows
  (recall 0.2809917355371901, precision 0.3541666666666667, F1
  0.31336405529953915), which is why its macro-F1 exceeds the forest's.
- `random_forest` is strong on frequent classes (F1: `punt` 0.9564,
  `field_goal` 0.8981, `qb_kneel` 0.7794 with precision 0.9705) but recovers
  only **2 of the 121** holdout `qb_spike` rows (recall 0.01652892561983471,
  precision 1.0, F1 0.032520325203252036). No calibration or threshold
  adjustment was applied.

Figures generated from the same payload: `results/comparison.png`
(accuracy vs macro-F1 per model), `results/random_forest_confusion.png`,
`results/per_class_f1.png`, and `results/random_forest_importance.png`
(impurity importances, descriptive only — biased toward high-cardinality
features). `random_forest_importance.png` is an inspection aid, not a causal
statement.

## Historical 2024 comparison (explicitly not comparable)

The predecessor 2024 course project reported micro-F1 figures around
0.75-0.77. Those numbers came from a different and unsafe protocol (a random
row-level split with transforms fitted outside the training fold, among other
differences) and are reported history, not measurements of this repository.
They are **not comparable** head-to-head with anything in the table above:
different split, different leakage posture, different code, different metric
aggregation (micro-F1 versus the macro-F1/accuracy reported here). The 2026
numbers in `results/metrics.json` are new, produced by this repository's
commands on the documented holdout.

## Limitations

- **Single holdout.** One chronological split (2017-2018) with no
  cross-validation and no test-set tuning; a different cutoff or season would
  give different numbers. The reported scores describe these games, not
  football in general.
- **Recorded label, not coach intent.** The target is the source's
  `play_type`. This study does not independently recover or verify the
  originally called play.
- **Uncalibrated probabilities.** `predict_proba` values come from an
  uncalibrated logistic regression and random forest; they are model
  probability estimates, not guarantees of calibrated confidence. No
  deployment or betting/decision claim is attached to them.
- **Rare classes are hard.** With 121 `qb_spike` holdout rows out of 66,965,
  a handful of correct predictions moves macro-F1 materially; the forest's
  2/121 spike recall is a measured failure mode, not an artifact to hide.
- **Approximate context.** `clock_running_proxy` is an explicit conservative
  approximation; missing numeric context is imputed with training medians
  inside the pipelines, and the source's own repeated/ambiguous rows are
  handled only by the documented explicit cleaning.
- **Data rights unresolved.** The source dataset's license is `Unknown` and
  upstream rights are not established; see `docs/provenance.md`. Results are
  aggregate measurements only, and no raw rows are published.
- **Not a product.** The saved bundle is a local research artifact; the CLI
  `predict` path serves the random forest for explicit hypothetical contexts
  only, and the static report under `docs/index.html` is a read-only page, not
  a prediction service.

## Environment of the recorded run

- Python 3.12.13, scikit-learn 1.5.2, numpy 1.26.4, pandas 2.2.3 (all four
  recorded in the run metadata). The figure and bundle dependencies
  (matplotlib 3.9.2, joblib 1.6.0) are pinned in `pyproject.toml` / `uv.lock`.
- Recorded `generated_utc`: 2026-10-03T03:28:44+00:00.
- Seeds: `seed=42` for split-independent estimators, forest
  `random_state=42`, `trees=200`, `jobs=2`.
