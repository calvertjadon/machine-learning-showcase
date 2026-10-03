# NFL play-by-play data (2009-2018)

This study classifies play types from pre-play context using the public
*NFL Play by Play 2009-2018 (v5)* CSV. This page names the file to obtain, how
to verify it, the schema and labels to expect, the training and evaluation
commands, and what this repository does not redistribute.

The data file is not included in this repository, so every user obtains a copy
from Kaggle. The full rights statement is in `provenance.md`.

## Source and version

The dataset is "Detailed NFL Play-by-Play Data 2009-2018", published on Kaggle
at <https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016>,
where its reference is `maxhorowitz/nflplaybyplay2009to2016`. Max Horowitz
published it, and the dataset description credits the compilation to Ron
Yurko, Sam Ventura, and Max Horowitz, the authors of
[nflscrapR](https://github.com/maksimhorowitz/nflscrapR), which scraped and
parsed the data from the NFL's API.

The file this project uses is distributed in Kaggle version 6, the current
version, which Kaggle's own metadata dates to 2018-12-22. Kaggle's version
history, from
<https://www.kaggle.com/api/v1/datasets/view/maxhorowitz/nflplaybyplay2009to2016>:

  | Kaggle version | Date | Notes |
  | --- | --- | --- |
  | 6 | 2018-12-22 | "Updated data with new & additional variables." |
  | 5 | 2018-03-17 | "Updated to include 2017 data." |
  | 4 | 2018-03-17 | "Data updated to include 2017 season." |
  | 3 | 2018-01-10 | "Updated to include all games including OT games." |
  | 2 | 2017-07-28 | "New version includes all data from 2009-2016 (mistake in previous upload only included 2009)." |
  | 1 | 2017-07-26 | "Initial release." |

The `(v5)` in the filename is the data-release label, not the Kaggle version
number. Kaggle version 5 contains only the 2009-2016 `(v3)` and 2009-2017
`(v4)` files. Reading the ZIP directory of both version bundles on 2026-10-02
confirmed this: version 6 contains `NFL Play by Play 2009-2016 (v3).csv`,
`NFL Play by Play 2009-2017 (v4).csv`, and
`NFL Play by Play 2009-2018 (v5).csv`.

Kaggle's metadata lists the license as `Unknown`, which means no license is
declared. Upstream, the nflscrapR package is CC0, but the companion data
repository declares no data license, and NFL.com's terms assert ownership of
content and data retrieved from its services. This project claims no rights in
the data and does not distribute it. Obtain the file yourself and assess the
terms that apply to you; see
[provenance.md](provenance.md#data-rights-detail).

## The file to download

| Property | Value |
| --- | --- |
| Filename (exact) | `NFL Play by Play 2009-2018 (v5).csv` |
| Format | comma-separated, UTF-8 with a byte-order mark (`EF BB BF`) |
| Size (uncompressed) | 700,397,316 bytes, about 668 MiB |
| SHA-256 | `33708680002e8e57eb42b1f41ff45b214fac8422952a89bb27acbeb18726dc90` |
| Data rows | 449,371 parsed data rows plus a header row, counted before cleaning; see [source data quality and cleaning](#source-data-quality-and-cleaning-of-the-verified-file) |
| Columns | 255 |
| Games | 2,526 |
| Date range | 2009-09-10 through 2018-12-17 |
| Seasons (parsed rows) | 2009: 41,880; 2010: 44,695; 2011: 45,017; 2012: 48,067; 2013: 45,248; 2014: 45,502; 2015: 43,257; 2016: 45,609; 2017: 48,060; 2018: 42,036 |

These row, season, and game counts come from parsing the file as CSV. Splitting
on lines or commas by hand can miscount, because quoted free text can contain
commas and line breaks.

The SHA-256 was established on 2026-10-02. The Kaggle version-6 download bundle
is a ZIP, 287,411,671 bytes at retrieval time, whose central directory records
the entry `NFL Play by Play 2009-2018 (v5).csv` as 700,397,316 bytes with
CRC-32 `cbaee00c`. Hashing a copy that matched that size and CRC-32 produced
the SHA-256 above. `nfl-showcase train` computes the SHA-256 of the file you
pass it and records the value in `artifacts/training.json`, and
`nfl-showcase evaluate` refuses to run when the data file no longer matches
that recorded value, so an older or modified release cannot be mixed in
silently.

The dataset page's prose description reads "...all the regular season plays
from the 2009-2016 NFL seasons... 356,768 rows and 100 columns". That text
describes an earlier upload and does not match the v5 file, so trust the
file-level checks above rather than the prose.

## Acquiring the file from the release

1. Obtain the dataset through Kaggle's standard flow. The website may require
   you to sign in, and Kaggle's CLI authenticates with API credentials; see
   <https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md>.
2. Open the dataset page at
   <https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016> and
   confirm the version selector is on Version 6, the default and current
   version.
3. Download the dataset bundle, a ZIP containing three CSV files.
4. Extract the member named exactly `NFL Play by Play 2009-2018 (v5).csv`.
   Do not use the `(v3)` or `(v4)` files, which cover fewer seasons and have
   different columns.
5. Verify before training:

   ```bash
   sha256sum "NFL Play by Play 2009-2018 (v5).csv"
   # expected: 33708680002e8e57eb42b1f41ff45b214fac8422952a89bb27acbeb18726dc90
   ```

   Also check that the size is 700,397,316 bytes and that the file starts with
   a UTF-8 byte-order mark followed by `play_id,game_id,...`.
6. Keep the file local and pass its path to the commands below. Do not commit
   it; `.gitignore` excludes `data/`, `*.csv`, and generated model files.

Kaggle also documents a CLI route, and the CLI authenticates with API
credentials:

```bash
kaggle datasets download -d maxhorowitz/nflplaybyplay2009to2016 --unzip -p data
```

The documented download options include `-f/--file`, `-p/--path`, `--unzip`,
and `--force`, but no version selector; see
<https://github.com/Kaggle/kaggle-api/blob/main/docs/datasets.md>. That is
fine here because the required file lives in the current version. The command
`kaggle datasets files maxhorowitz/nflplaybyplay2009to2016` lists the files
before you download them, and the website's version selector browses or pins
versions explicitly.

## Expected schema

The pipeline reads the raw file with `encoding="utf-8-sig"` and
`usecols=RAW_COLUMNS`, so it selects columns by name and ignores the rest. All
23 read columns exist in the v5 header:

```text
game_id, play_id, game_date, home_team, away_team, posteam, defteam,
yardline_100, quarter_seconds_remaining, half_seconds_remaining, qtr, down,
ydstogo, posteam_timeouts_remaining, defteam_timeouts_remaining,
score_differential, play_type, yards_gained, desc, qb_spike, quarter_end,
penalty, timeout
```

- The file begins `play_id,game_id,home_team,away_team,posteam,...`, with the
  BOM before `play_id`. Free-text fields such as `desc` contain commas and
  are quoted, so the file needs a real CSV parser rather than naive comma
  splitting. Row counts likewise come from the parser, because quoted free
  text can contain line breaks.
- The `game_id` column is a 10-digit game identifier, and `play_id` is the
  play number within the game. The pair `(game_id, play_id)` is not unique
  in this file: the source has 2,437 repeated identifier rows and 22
  conflicting identifiers, as described under
  [source data quality and cleaning](#source-data-quality-and-cleaning-of-the-verified-file).
  The explicit `clean_raw_plays` step makes the pair unique before
  `prepare_plays`, which still rejects any shared identifier on purpose.
- `game_date` uses the `YYYY-MM-DD` format, and each game has a single date.
- Team codes are nflscrapR-era abbreviations, and preparation normalizes the
  relocations `OAK->LV`, `SD->LAC`, `STL/LA->LAR`, and `JAC->JAX`.
- Missing values occur in the file, for example `down` on some rows,
  `NA`-marked `play_type` values, and missing previous-play outcomes.
  Preparation preserves them, and a row with a supported target label is kept
  even when its numeric context is missing. The only row-dropping decisions
  are explicit and listed here: the exact-repeat collapse and the opt-in
  whole-game exclusion in `clean_raw_plays`, plus the target-label filter in
  preparation. Imputation is fitted on training rows only; see
  [features.md](features.md#missing-values-and-the-imputation-boundary).
- The remaining columns include player IDs and names, free-text
  descriptions, and expected-points/win-probability model outputs. They are
  never model features and never leave `prepare_plays` as raw columns. The
  full audit is in [features.md](features.md).

## Source data quality and cleaning of the verified file

The v5 file is not a perfectly canonical table. On 2026-10-02 a read-only
diagnosis of the exact file identified by the size and SHA-256 above found the
counts below. The evidence is recorded here so that every run does not have to
re-derive it.

| Observation | Count |
| --- | --- |
| Parsed data rows | 449,371 |
| Repeated `(game_id, play_id)` rows, counting extra rows beyond the first row of each shared identifier | 2,437 |
| Rows fully identical across the 23 selected columns, which are the exact repeats | 2,393 |
| Rows after collapsing exact repeats | 446,978 |
| Conflicting identifiers remaining after collapse | 22 |
| Games containing those conflicts | 1, dated 2011-12-04 |
| Rows of that game after exact repeat collapse | 268 |
| Retained raw rows when that whole game is excluded | 446,710 |

The collapse removes exactly those 2,393 repeats. Afterward, 22 identifiers
still have more than one remaining distinct selected record, and all of them
lie in the single game dated 2011-12-04. Only `posteam_timeouts_remaining` and
`defteam_timeouts_remaining` differ within those conflict groups. The file
does not record which value was true before the snap, and row order does not
reliably tell which record came first, so the pipeline adds no keep-last rule,
no averaging, and no zero fill.

Instead, `clean_raw_plays` handles those rows explicitly. The full semantics
are in
[features.md](features.md#cleaning-repeated-and-ambiguous-rows-with-clean_raw_plays):

- Rows identical across all 23 selected columns collapse to the first
  occurrence, which removes 2,393 rows from this file.
- The 22 conflicting identifiers make their game ambiguous. With the default
  `exclude_ambiguous_games=False`, the source is rejected with an actionable
  `ValueError` that names the counts and the opt-in.
- The documented real-data command below opts in with
  `--exclude-ambiguous-games`. That drops every row of the 2011-12-04 game,
  268 rows after collapse including its non-conflicting plays, before any
  previous-play context is derived, so context and whole-game evaluation
  never cross a partially removed game.

Collapse plus whole-game exclusion leaves 446,710 raw source rows. The
`clean_raw_plays` summary reports this arithmetic through `raw_rows`,
`exact_duplicate_rows_removed`, `ambiguous_id_groups`,
`ambiguous_games_excluded`, `ambiguous_rows_removed`, and `retained_rows`. The
SHA-256 above belongs to the full source file as downloaded, and cleaning does
not change it.

## Targets

The label comes from `play_type`, and the pipeline keeps exactly six labels in
sorted order: `field_goal`, `pass`, `punt`, `qb_kneel`, `qb_spike`, `run`.

The table below counts raw rows in the retrieved v5 file, not prepared rows.
The counts come from parsing the file as CSV.

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

The table lists all ten distinct `play_type` values in the file, and they sum
to its 449,371 parsed rows. These are raw file counts from before cleaning, so
they include the 2,393 exactly repeated records and the 22 conflicting
identifiers. The six target labels cover 357,580 of the raw rows.

After `clean_raw_plays` collapses exact repeats, and drops the ambiguous game
when the caller opts in, `prepare_plays` derives the previous-play context
from every remaining original row and then keeps only the rows whose
`play_type` is one of the six target labels. That label filter and the
cleaning decisions above are the only filters that drop rows. A
target-labelled row is retained even when its numeric context is missing, and
those gaps stay `NaN` in the prepared frame and are imputed during training on
training rows only, as described under
[missing values and the imputation boundary](features.md#missing-values-and-the-imputation-boundary).
The label distribution is strongly imbalanced, which is why evaluation reports
macro-F1 and per-class support alongside accuracy. Prepared counts depend on
the cleaning choice: the 2,393 collapses always apply, and the extra 268-row
whole-game exclusion applies only with `--exclude-ambiguous-games`.
`nfl-showcase train` records the counts for each run in
`artifacts/training.json` rather than guessing or hard-coding them.

## Train/evaluate split

`chronological_split` defaults to `holdout_start=2017-09-01`. Training data is
every game dated strictly before 2017-09-01, which covers the 2009-2016
seasons. Test data is every game dated on or after 2017-09-01, which covers
the 2017 and 2018 seasons. No game appears in both partitions, and no game is
split across the cutoff.

This is a temporal, game-separated holdout rather than a random row split, so
it measures generalization to later games and prevents row-level and same-game
leakage.

## Commands

Training fits the models and writes the run manifest and the local model
bundle:

```bash
uv run nfl-showcase train \
  --data "/path/to/NFL Play by Play 2009-2018 (v5).csv" \
  --exclude-ambiguous-games \
  --output-dir artifacts \
  --holdout-start 2017-09-01 \
  --seed 42 --trees 200 --jobs 2
```

`--exclude-ambiguous-games` is required to process this source. Without it,
`clean_raw_plays` rejects the 22 conflicting identifiers in the 2011-12-04
game with an actionable error. The flag makes the whole-game exclusion
explicit, and it runs before `prepare_plays`, so no previous-play context is
stitched across deleted plays. The flag defaults to off, and a source with no
ambiguity runs unchanged without it.

Evaluation requires the data file to hash-match the recorded run:

```bash
uv run nfl-showcase evaluate \
  --data "/path/to/NFL Play by Play 2009-2018 (v5).csv" \
  --model artifacts/models.joblib \
  --output-dir results
```

- `train` writes `artifacts/models.joblib`, which stays local and is never
  published, and `artifacts/training.json`. The manifest records the source
  filename rather than a private path, the byte size, the full SHA-256 of the
  downloaded file before cleaning, the scikit-learn and Python versions, the
  cutoff, the seed, the feature columns, the `exclude_ambiguous_games` choice,
  the `source_cleaning` summary returned by `clean_raw_plays`, train and test
  row and game counts, the minimum and maximum dates, the label counts, and
  the forest parameters.
- `evaluate` writes `results/metrics.json` and the figures for model
  comparison, the random-forest confusion matrix, and per-class F1. It aborts
  if the data file's SHA-256 does not match the recorded value. It then
  re-runs the cleaning with the recorded `exclude_ambiguous_games` value and
  aborts if the recomputed `source_cleaning` summary differs from the recorded
  one.
- `uv run nfl-showcase predict --model artifacts/models.joblib --input IN.json`
  scores single pre-play context rows. All feature columns are required, and
  the caller is expected to supply honest previous-play context; see the
  README.

## Publication and privacy

- The raw CSV is not committed, published, or redistributed. `.gitignore`
  excludes `data/`, `*.csv`, and generated model files under `artifacts/`,
  including `*.joblib` and `*.pkl`.
- Derived reports contain only aggregates, so they carry no rows, no play
  descriptions, no player names or IDs, and no absolute local paths.
- Model bundles are local artifacts. Do not load `.joblib` or pickle files
  from untrusted sources; scikit-learn bundles are reliable only with a
  compatible scikit-learn version.
- The dataset license is undeclared, Kaggle lists it as `Unknown`, and the
  upstream rights are unresolved, so this repository cannot grant anyone
  rights to the data. Any use of the dataset is between you, Kaggle, and the
  upstream rights holders. See
  [provenance.md](provenance.md#data-rights-detail).

## Primary sources

- Kaggle dataset page: <https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016>
- Kaggle dataset metadata API (license `Unknown`, version history): <https://www.kaggle.com/api/v1/datasets/view/maxhorowitz/nflplaybyplay2009to2016>
- nflscrapR scraper (`License: CC0` in `DESCRIPTION`): <https://github.com/maksimhorowitz/nflscrapR>
- nflscrapR-data repository (no license declared): <https://github.com/ryurko/nflscrapR-data>
- nflWAR paper (expected points / win probability models): <https://arxiv.org/abs/1802.00998>
- NFL.com Terms and Conditions: <https://www.nfl.com/legal/terms/>
- Kaggle Terms of Use: <https://www.kaggle.com/terms>
- Kaggle CLI documentation: <https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md>
