# Data: NFL play-by-play (2009-2018)

The NFL showcase classifies play types from pre-play context using the public
*NFL Play by Play 2009-2018 (v5)* CSV. This page documents exactly which file
to obtain, how to verify it, what schema and labels to expect, how to run the
training and evaluation commands, and what this repository does not
redistribute.

The data file is **not** included in this repository; every user obtains
their own copy. See `provenance.md` for the full rights statement.

## Source and version

- **Dataset**: "Detailed NFL Play-by-Play Data 2009-2018",
  <https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016>
  (Kaggle reference `maxhorowitz/nflplaybyplay2009to2016`).
- **Published by**: Max Horowitz. The dataset description credits compilation
  to Ron Yurko, Sam Ventura, and Max Horowitz, the authors of
  [nflscrapR](https://github.com/maksimhorowitz/nflscrapR), which scraped and
  parsed the data from the NFL's API.
- **Kaggle dataset version**: the file this project uses is distributed in
  **version 6**, the current version (last updated 2018-12-22 per Kaggle's
  own metadata). Kaggle's version history, from
  <https://www.kaggle.com/api/v1/datasets/view/maxhorowitz/nflplaybyplay2009to2016>:

  | Kaggle version | Date | Notes |
  | --- | --- | --- |
  | 6 | 2018-12-22 | "Updated data with new & additional variables." |
  | 5 | 2018-03-17 | "Updated to include 2017 data." |
  | 4 | 2018-03-17 | "Data updated to include 2017 season." |
  | 3 | 2018-01-10 | "Updated to include all games including OT games." |
  | 2 | 2017-07-28 | "New version includes all data from 2009-2016 (mistake in previous upload only included 2009)." |
  | 1 | 2017-07-26 | "Initial release." |

- **The `(v5)` in the filename is the data-release label, not the Kaggle
  version number.** Kaggle version 5 contains only the 2009-2016 `(v3)` and
  2009-2017 `(v4)` files. This was checked on 2026-10-02 by reading the ZIP
  directory of both version bundles: version 6 contains
  `NFL Play by Play 2009-2016 (v3).csv`,
  `NFL Play by Play 2009-2017 (v4).csv`, and
  `NFL Play by Play 2009-2018 (v5).csv`.
- **License**: Kaggle's metadata lists the license as `Unknown` (no license
  declared). Upstream, the nflscrapR package is CC0 but the companion data
  repository declares no data license, and NFL.com's terms assert ownership
  of content and data retrieved from its services. This project claims no
  rights in the data and does not distribute it. Obtain the file yourself
  and assess the terms that apply to you; see
  [provenance.md](provenance.md#data-rights-detail).

## The file to download

| Property | Value |
| --- | --- |
| Filename (exact) | `NFL Play by Play 2009-2018 (v5).csv` |
| Format | comma-separated, UTF-8 with BOM (`EF BB BF`) |
| Size (uncompressed) | 700,397,316 bytes (about 668 MiB) |
| SHA-256 | `33708680002e8e57eb42b1f41ff45b214fac8422952a89bb27acbeb18726dc90` |
| Data rows | 449,371 parsed data rows plus one header row (raw file count, before cleaning; see "Source data quality and cleaning") |
| Columns | 255 |
| Games | 2,526 |
| Date range | 2009-09-10 through 2018-12-17 |
| Seasons (parsed rows) | 2009: 41,880; 2010: 44,695; 2011: 45,017; 2012: 48,067; 2013: 45,248; 2014: 45,502; 2015: 43,257; 2016: 45,609; 2017: 48,060; 2018: 42,036 |

The row, season, and game counts above come from parsing the file as CSV.
Naive line or comma splitting can miscount it, because quoted free text can
contain commas or line breaks.

How the SHA-256 was established (2026-10-02): the Kaggle version-6 download
bundle is a ZIP (287,411,671 bytes at retrieval time) whose central directory
records the entry `NFL Play by Play 2009-2018 (v5).csv` as 700,397,316 bytes
with CRC-32 `cbaee00c`. A copy matching that size and CRC-32 was hashed; the
SHA-256 above is its value. `nfl-showcase train` computes the SHA-256 of the
file you pass it and records it in `artifacts/training.json`, and
`nfl-showcase evaluate` refuses to run when the data file no longer matches
that recorded value, so an older or modified release cannot be mixed in
silently.

Note: the dataset page's prose description ("...all the regular season plays
from the 2009-2016 NFL seasons... 356,768 rows and 100 columns") describes an
earlier upload and does not match the v5 file. Trust the file-level checks
above, not the prose.

## Acquisition (manual, matching the release)

1. Obtain the dataset through Kaggle's standard flow: the website may require
   you to sign in, and Kaggle's CLI authenticates with API credentials; see
   <https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md>.
2. Open the dataset page:
   <https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016>.
   Confirm the version selector is on **Version 6** (the default/current
   version).
3. Download the dataset bundle (a ZIP containing three CSV files).
4. Extract the member named exactly `NFL Play by Play 2009-2018 (v5).csv`.
   Do not use the `(v3)` or `(v4)` files: they cover fewer seasons and have
   different columns.
5. Verify before training:

   ```bash
   sha256sum "NFL Play by Play 2009-2018 (v5).csv"
   # expected: 33708680002e8e57eb42b1f41ff45b214fac8422952a89bb27acbeb18726dc90
   ```

   Also check the size (700,397,316 bytes) and that the file starts with a
   UTF-8 BOM followed by `play_id,game_id,...`.
6. Keep the file local and pass its path to the commands below. Do **not**
   commit it: `.gitignore` excludes `data/`, `*.csv`, and generated model
   files.

Optional CLI route (documented by Kaggle; the CLI authenticates with API
credentials):

```bash
kaggle datasets download -d maxhorowitz/nflplaybyplay2009to2016 --unzip -p data
```

The documented download options include `-f/--file`, `-p/--path`, `--unzip`,
and `--force`, but no version selector (see
<https://github.com/Kaggle/kaggle-api/blob/main/docs/datasets.md>); that is
fine here because the required file lives in the current version. To inspect
files before downloading, `kaggle datasets files
maxhorowitz/nflplaybyplay2009to2016` lists them. To browse or pin versions
explicitly, use the website's version selector.

## Expected schema

The raw file is read with `encoding="utf-8-sig"` and
`usecols=RAW_COLUMNS`; the pipeline selects columns **by name** and ignores
the rest. All 23 read columns exist in the v5 header:

```text
game_id, play_id, game_date, home_team, away_team, posteam, defteam,
yardline_100, quarter_seconds_remaining, half_seconds_remaining, qtr, down,
ydstogo, posteam_timeouts_remaining, defteam_timeouts_remaining,
score_differential, play_type, yards_gained, desc, qb_spike, quarter_end,
penalty, timeout
```

- The file begins `play_id,game_id,home_team,away_team,posteam,...` (the BOM
  precedes `play_id`). Free-text fields such as `desc` contain commas and
  are quoted, so the file must be parsed with a real CSV parser, not naive
  comma splitting; row counts likewise come from the parser, because quoted
  free text can contain line breaks.
- Identifiers: `game_id` is a 10-digit game identifier and `play_id` is the
  play number within the game. The pair `(game_id, play_id)` is *not* unique
  in this file: 2,437 repeated identifier rows and 22 conflicting identifiers
  were observed (see "Source data quality and cleaning" below). The pair must
  be made unique by the explicit `clean_raw_plays` step before
  `prepare_plays`, which deliberately still rejects any shared identifier.
- `game_date` is `YYYY-MM-DD`, and each game has a single date.
- Team codes are nflscrapR-era abbreviations; preparation normalizes
  relocations `OAK->LV`, `SD->LAC`, `STL/LA->LAR`, and `JAC->JAX`.
- Missing values occur (for example `down` on some rows, `NA`-marked
  `play_type` values, missing previous-play outcomes). Preparation preserves
  them: a row with a supported target label is kept even when its numeric
  context is missing. The only row-dropping decisions are explicit and
  audited — the exact-repeat collapse and the opt-in whole-game exclusion in
  `clean_raw_plays` (see below), plus preparation's target-label filter;
  imputation is fitted on training rows only (see
  [features.md](features.md#missing-values-and-the-imputation-boundary)).
- The remaining columns include player IDs and names, free-text
  descriptions, and expected-points/win-probability model outputs. They are
  never model features and never leave `prepare_plays` as raw columns; see
  [features.md](features.md) for the full audit.

## Source data quality and cleaning (verified file)

The v5 file is not a perfectly canonical table. A read-only diagnosis of the
exact file identified by the size and SHA-256 above (2026-10-02; the evidence
is recorded here rather than re-derived by every run) found:

| Observation | Count |
| --- | --- |
| Parsed data rows | 449,371 |
| Repeated `(game_id, play_id)` rows (extra rows beyond the first row of each shared identifier) | 2,437 |
| Rows fully identical across the 23 selected columns (exact repeats) | 2,393 |
| Rows after collapsing exact repeats | 446,978 |
| Conflicting identifiers remaining after collapse | 22 |
| Games containing those conflicts | 1 (dated 2011-12-04) |
| Rows of that game after exact repeat collapse | 268 |
| Retained raw rows when that whole game is excluded | 446,710 |

The 2,393 repeats are exactly what the collapse removes; afterwards 22
identifiers (all in the single 2011-12-04 game) still have more than one
remaining distinct selected record. In those conflict groups only
`posteam_timeouts_remaining` and `defteam_timeouts_remaining` differ; the
file does not record which value was true pre-snap, and row order is not
reliable evidence. The pipeline therefore invents no keep-last rule, no
averaging, and no zero fill.

Instead, `clean_raw_plays` is the explicit, audited contract (full semantics
in [features.md](features.md#clean_raw_plays-explicit-handling-of-repeated-and-ambiguous-rows)):

- rows identical across all 23 selected columns collapse, keeping the first
  occurrence (2,393 rows for this file);
- the 22 conflicting identifiers make their game ambiguous; the default
  `exclude_ambiguous_games=False` rejects the source with an actionable
  `ValueError` naming the counts and the opt-in;
- the documented real-data command below opts in with
  `--exclude-ambiguous-games`, which drops *every* row of the 2011-12-04 game
  (268 rows after collapse, including its non-conflicting plays) before any
  previous-play context is derived, so context and whole-game evaluation can
  never cross a partially removed game.

Collapse plus whole-game exclusion leaves 446,710 raw source rows; the
`clean_raw_plays` summary reports this arithmetic (`raw_rows`,
`exact_duplicate_rows_removed`, `ambiguous_id_groups`,
`ambiguous_games_excluded`, `ambiguous_rows_removed`, `retained_rows`). The
SHA-256 above is of the full source file as downloaded and is unchanged by
cleaning.

## Targets

The label is derived from `play_type`; the pipeline keeps exactly these six
labels (sorted): `field_goal`, `pass`, `punt`, `qb_kneel`, `qb_spike`, `run`.

Raw `play_type` counts in the retrieved v5 file, counted by parsing the file
as CSV (raw row counts, not prepared counts):

| play_type | rows |
| --- | --- |
| pass | 186,677 |
| run | 132,692 |
| punt | 23,914 |
| field_goal | 9,777 |
| qb_kneel | 3,830 |
| qb_spike | 690 |
| *(dropped)* no_play | 42,431 |
| *(dropped)* kickoff | 25,552 |
| *(dropped)* `NA` (missing) | 12,874 |
| *(dropped)* extra_point | 10,934 |

These are all ten distinct `play_type` values in the file, and they sum to
its 449,371 parsed rows. They are raw file counts *before cleaning*: they
include the 2,393 exactly repeated records and the 22 conflicting identifiers;
the six target labels cover 357,580 of those raw rows. After `clean_raw_plays`
has collapsed exact repeats (and, when opted in, dropped the ambiguous game),
`prepare_plays` derives the previous-play context from every remaining original
row and then keeps exactly the rows whose `play_type` is one of the six target
labels. That label filter and the audited cleaning decisions are the only
row-dropping filters, and a target-labelled row is retained even when its
numeric context is missing. Those gaps stay `NaN` in the prepared frame and
are imputed during training on training rows only (see
[features.md](features.md#missing-values-and-the-imputation-boundary)).
The distribution is strongly imbalanced, which is why evaluation reports
macro-F1 and per-class support alongside accuracy. Prepared counts depend on
the cleaning choice (always the 2,393 collapses; the extra 268-row whole-game
exclusion only with `--exclude-ambiguous-games`) and are recorded for each
run by `nfl-showcase train` in `artifacts/training.json` (label counts); they
are not guessed or hard-coded here.

## Train/evaluate split

`chronological_split` defaults to `holdout_start=2017-09-01`:

- **Train**: games dated strictly before 2017-09-01 (the 2009-2016 seasons).
- **Test**: games dated on or after 2017-09-01 (the 2017-2018 seasons).
- No game appears in both partitions and no game is split across the cutoff.

This is a temporal, game-separated holdout rather than a random row split: it
measures generalization to later games and prevents row-level and same-game
leakage.

## Commands

Train (fits the models, writes the run manifest and the local model bundle):

```bash
uv run nfl-showcase train \
  --data "/path/to/NFL Play by Play 2009-2018 (v5).csv" \
  --exclude-ambiguous-games \
  --output-dir artifacts \
  --holdout-start 2017-09-01 \
  --seed 42 --trees 200 --jobs 2
```

`--exclude-ambiguous-games` is required to process this source: without it,
`clean_raw_plays` rejects the 22 conflicting identifiers in the 2011-12-04
game with an actionable error. The flag makes that whole-game exclusion
explicit; it runs before `prepare_plays`, so no previous-play context is
stitched across deleted plays. (The flag defaults to off, and a source with
no ambiguity runs unchanged without it.)

Evaluate (requires the data file to hash-match the recorded run):

```bash
uv run nfl-showcase evaluate \
  --data "/path/to/NFL Play by Play 2009-2018 (v5).csv" \
  --model artifacts/models.joblib \
  --output-dir results
```

- `train` writes `artifacts/models.joblib` (local only, never published) and
  `artifacts/training.json`, which records the source filename (not a private
  path), byte size, full SHA-256 (of the downloaded file, before cleaning),
  scikit-learn and Python versions, cutoff, seed, feature columns, the
  `exclude_ambiguous_games` choice, the `source_cleaning` summary returned by
  `clean_raw_plays`, train/test row and game counts, min/max dates, label
  counts, and forest parameters.
- `evaluate` writes `results/metrics.json` plus the figures (model
  comparison, random-forest confusion matrix, per-class F1). It aborts if the
  data file's SHA-256 does not match the recorded value, re-runs the cleaning
  with the recorded `exclude_ambiguous_games` value, and aborts if the
  recomputed `source_cleaning` summary differs from the recorded one.
- `uv run nfl-showcase predict --model artifacts/models.joblib --input IN.json`
  scores pre-play context rows (all feature columns required, honest
  previous-play context expected; see the README).

## Publication and privacy

- The raw CSV is **not** committed, published, or redistributed.
  `.gitignore` excludes `data/`, `*.csv`, and generated model files
  (`artifacts/`, `*.joblib`, `*.pkl`).
- Derived reports contain aggregates only: no rows, no play descriptions, no
  player names or IDs, and no absolute local paths.
- Model bundles are local artifacts. Do not load `.joblib`/pickle files from
  untrusted sources; scikit-learn bundles are only reliable with a compatible
  scikit-learn version.
- Because the dataset license is undeclared (`Unknown`) and the upstream
  rights are unresolved, this repository cannot grant anyone rights to the
  data. Use of the dataset is between you, Kaggle, and the upstream rights
  holders. See [provenance.md](provenance.md#data-rights-detail).

## Primary sources

- Kaggle dataset page: <https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016>
- Kaggle dataset metadata API (license `Unknown`, version history): <https://www.kaggle.com/api/v1/datasets/view/maxhorowitz/nflplaybyplay2009to2016>
- nflscrapR scraper (`License: CC0` in `DESCRIPTION`): <https://github.com/maksimhorowitz/nflscrapR>
- nflscrapR-data repository (no license declared): <https://github.com/ryurko/nflscrapR-data>
- nflWAR paper (expected points / win probability models): <https://arxiv.org/abs/1802.00998>
- NFL.com Terms and Conditions: <https://www.nfl.com/legal/terms/>
- Kaggle Terms of Use: <https://www.kaggle.com/terms>
- Kaggle CLI documentation: <https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md>
