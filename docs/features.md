# Feature engineering and the evaluation split

This document describes how `nfl_showcase.features` turns raw play-by-play
rows into a causal, pre-snap feature frame, and how `chronological_split`
divides that frame into a game-separated holdout from future seasons. It lists
every prediction input with the timing argument that justifies it, and lists
everything rejected with the reason. The models and the CLI rely on this
contract.

The raw data is the public *NFL Play by Play 2009-2018 (v5)* CSV, which holds
about 450k plays across 255 columns and uses UTF-8 with a BOM. The source file
is not bundled in this repository, so see [data.md](data.md) for download and
rights information. The verified file also contains repeated records and one
game whose play identifiers carry conflicting pre-snap values.
`clean_raw_plays` handles those rows explicitly, and the exact diagnosis is
recorded under
[source data quality and cleaning](data.md#source-data-quality-and-cleaning-of-the-verified-file).

## Public interface

| Name | Contents |
| --- | --- |
| `RAW_COLUMNS` | The 23 raw columns read from the CSV, covering identifiers, date, teams, field position, clock, down and distance, timeouts, score, play type, and the columns read to audit the previous play: `yards_gained`, `desc`, `qb_spike`, `quarter_end`, `penalty`, and `timeout`. |
| `NUMERIC_FEATURES` | `yardline_100`, `quarter_seconds_remaining`, `half_seconds_remaining`, `qtr`, `down`, `ydstogo`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`, `score_differential`, `previous_yards_gained`, `clock_running_proxy`. |
| `CATEGORICAL_FEATURES` | `home_team`, `away_team`, `posteam`, `defteam`, `previous_play_type`. |
| `FEATURE_COLUMNS` | `NUMERIC_FEATURES + CATEGORICAL_FEATURES`. These are the complete model input vocabulary in this exact order. |
| `TARGET_LABELS` | `field_goal`, `pass`, `punt`, `qb_kneel`, `qb_spike`, `run`, in sorted order. |
| `prepare_plays(raw)` | Returns `game_id`, `play_id`, `game_date`, `play_type` plus `FEATURE_COLUMNS` only. |
| `clean_raw_plays(raw, *, exclude_ambiguous_games=False)` | Cleans the raw frame before preparation. It collapses rows repeated across the 23 selected `RAW_COLUMNS` and, when the caller opts in, drops every row of the games affected by conflicting identifiers. It returns `(cleaned, summary)`, and the summary holds the counts. |
| `chronological_split(prepared, holdout_start="2017-09-01")` | Returns `(train, test)`. Train dates fall strictly before the cutoff, test dates fall on or after it, and no game appears in both. |

Read the raw CSV with `usecols=RAW_COLUMNS` and `encoding="utf-8-sig"`. The
BOM is why the encoding matters. Without `utf-8-sig`, the first column would
arrive as `﻿play_id` with the BOM attached, and `prepare_plays` would reject
the frame for a missing required column, which is the correct behavior.

## Cleaning repeated and ambiguous rows with `clean_raw_plays`

The verified source is not a perfectly canonical table.
[data.md](data.md#source-data-quality-and-cleaning-of-the-verified-file)
records the exact read-only diagnosis of the file identified by the SHA-256
listed there: 2,437 repeated `(game_id, play_id)` rows, of which 2,393 are
identical across all 23 selected columns. Collapsing the repeats leaves
446,978 rows and 22 identifiers whose remaining records still disagree. All 22
are in a single game dated 2011-12-04, which has 268 rows after the collapse.
Only `posteam_timeouts_remaining` and `defteam_timeouts_remaining` differ
within those groups. The source does not record which value was true before
the snap, and row order does not tell which record came first.

`prepare_plays` rejects shared identifiers outright, so callers working with
the real source run this step first:

```python
cleaned, summary = clean_raw_plays(raw, exclude_ambiguous_games=True)
prepared = prepare_plays(cleaned)
```

All cleaning semantics are defined only in the selected 23 `RAW_COLUMNS`:

1. The function validates the input frame. Missing required columns, an empty
   frame, missing or blank identifiers, missing or unparseable dates, and a
   game with conflicting dates all raise `ValueError`. Invalid-source errors
   are never hidden.
2. Exact repeats collapse. A record repeated across every selected column is a
   repeated row, and all but the first occurrence are dropped. No other
   equality is considered: equal identifiers alone never justify deleting a
   row, and there is no keep-last rule.
3. Ambiguity is detected after the collapse. A `(game_id, play_id)` shared by
   more than one row necessarily disagrees on at least one selected value,
   which the source alone cannot resolve, so the function invents no ordering,
   averaging, or fill rule.
4. Residual ambiguity is rejected by default. With
   `exclude_ambiguous_games=False`, the function raises an actionable
   `ValueError` that reports the number of conflicting identifier groups and
   the number of affected games and names the opt-in flag. It does not echo
   the identities of affected games.
5. The opt-in drops whole games. With `exclude_ambiguous_games=True`, every
   row of every affected game is dropped, not just the conflicting plays, so
   previous-event context and whole-game evaluation cannot cross a partially
   removed game. Rows and dates of unaffected games are preserved untouched.

The returned frame contains exactly the 23 selected columns in the original
row order, keeps the first occurrence, and has a fresh `RangeIndex`. The
summary has exactly these keys, each a plain Python `int`:

| Key | Meaning |
| --- | --- |
| `raw_rows` | Rows in `raw` before any cleaning. |
| `exact_duplicate_rows_removed` | Repeated records collapsed. |
| `ambiguous_id_groups` | `(game_id, play_id)` groups still disagreeing after collapse (`0` if unambiguous). |
| `ambiguous_games_excluded` | Distinct games dropped by the opt-in (`0` when none). |
| `ambiguous_rows_removed` | Rows dropped by that whole-game exclusion, counted *after* repeat collapse. |
| `retained_rows` | Rows returned in `cleaned`. |

Excluded game identities do not appear in the summary, and a result emptied by
exclusion raises `ValueError`. For the verified source, the counts are 2,393
collapsed exact repeats, and the caller opting in excludes one game and 268
rows, which leaves 446,710 retained raw rows. `data.md` states the same facts
from the source side. Cleaning does not change the source SHA-256, which is
the hash of the full downloaded file.

## `prepare_plays` pipeline

The steps are ordered so that no information leaks backward in time.

1. Every `RAW_COLUMNS` name must be present, and extra columns are ignored. A
   missing column raises `ValueError`.
2. An empty raw frame raises `ValueError`.
3. `game_id` and `play_id` must be non-missing and non-blank. Blank includes
   the `NA` tokens the source file uses for missing values. Whole numeric
   identifiers are normalized to `int64` so that ordering and duplicate
   detection are numeric. Non-numeric identifiers are kept as trimmed
   strings, and fractional numeric identifiers are rejected.
4. `game_date` is parsed with `errors="coerce"`. Missing or unparseable values
   raise `ValueError`, valid values are normalized to midnight, and dates are
   compared at date granularity.
5. Duplicate `(game_id, play_id)` pairs raise `ValueError`. That strictness is
   deliberate: the real source is cleaned with `clean_raw_plays` before
   preparation rather than resolved inside the pipeline.
6. Any game whose rows disagree on `game_date` raises `ValueError`, because a
   game is a single calendar event and must not span a split.
7. `play_type` and the four team columns are whitespace-trimmed, and blank
   text becomes a missing value. Relocated franchises are mapped to their
   current abbreviation: `OAK→LV`, `SD→LAC`, `STL→LAR`, `LA→LAR`, `JAC→JAX`.
8. All original rows are stable-sorted by `(game_id, play_id)` before any
   `shift` and before any label filtering. This ordering is what makes the
   previous-play features causal and reproducible.
9. Every derived column is a `groupby("game_id").shift(1)` of the ordered
   rows, so context never crosses a game boundary. The first play of each game
   has no previous row.
10. The clock-running proxy is derived from the previous row only, as
    described below.
11. Rows are filtered to `TARGET_LABELS` after the previous-play context has
    been derived from every original row, so the first target play after a
    kickoff, punt, penalty, or timeout still sees that context.
12. Only `game_id`, `play_id`, `game_date`, `play_type`, and
    `FEATURE_COLUMNS` are returned. Raw descriptions and outcome columns never
    leave the function.

## What each predictor is and when it is known

Timing is stated relative to the snap of the play whose `play_type` is being
predicted. A pre-snap value is part of the game state a coach can observe
before calling the play. A previous-play value is computed from the
immediately preceding row of the same game after the global sort.

| Feature | Source | Timing | Notes |
| --- | --- | --- | --- |
| `yardline_100` | `yardline_100` | Pre-snap | Distance from the opponent's goal line in yards, from 0 to 100. |
| `quarter_seconds_remaining` | `quarter_seconds_remaining` | Pre-snap | Seconds left in the current quarter. |
| `half_seconds_remaining` | `half_seconds_remaining` | Pre-snap | Seconds left in the current half. |
| `qtr` | `qtr` | Pre-snap | Period, 1 through 5; 5 is overtime. |
| `down` | `down` | Pre-snap | Current down. Missing values are kept, for example on two-point tries. |
| `ydstogo` | `ydstogo` | Pre-snap | Yards to the first down, or to the goal. |
| `posteam_timeouts_remaining` | `posteam_timeouts_remaining` | Pre-snap | Timeouts left for the possession team, with missing values kept. |
| `defteam_timeouts_remaining` | `defteam_timeouts_remaining` | Pre-snap | Timeouts left for the defence, with missing values kept. |
| `score_differential` | `score_differential` | Pre-snap | Possession-team score minus opponent score at the snap. |
| `previous_yards_gained` | `yards_gained` shifted within the game | Previous play | Outcome of the immediately preceding play of the same game. A kickoff or a missing value yields `NaN`. |
| `clock_running_proxy` | derived `int8` 0/1 | Previous play | A conservative estimate that the game clock kept running after the previous play, and 0 on a game's first row. |
| `home_team` | `home_team` | Pre-snap | Normalized abbreviation. |
| `away_team` | `away_team` | Pre-snap | Normalized abbreviation. |
| `posteam` | `posteam` | Pre-snap | Possession team as a normalized abbreviation. |
| `defteam` | `defteam` | Pre-snap | Defence team as a normalized abbreviation. |
| `previous_play_type` | `play_type` shifted within the game | Previous play | Raw previous-play type, such as `kickoff`, `punt`, `no_play`, a special-teams label, or any target label. It is `unknown` when the previous play's type is missing or blank or when the row opens a game. |

The identifier columns `game_id`, `play_id`, and `game_date` are returned for
partitioning and reporting only and are not model inputs. `play_type` is the
target label.

## The clock-running proxy in detail

Possession changes, penalties, spikes, timeouts, and boundary plays all stop
the clock, but the raw table does not record a trustworthy "clock was running"
indicator, so the module derives a proxy from the previous row. The proxy is
conservative and returns `1` only when

```text
has previous row
and current qtr is known
and previous qtr == current qtr
and previous play_type in {run, pass, qb_kneel}
and previous qb_spike == 0
and previous quarter_end == 0
and previous penalty == 0
and previous timeout == 0
and previous desc does not match (case-insensitive)
    incomplete | out of bounds | two-minute | end of quarter | timeout
```

and `0` otherwise. This has these consequences:

* A game's first row is always `0`.
* The first play after a kickoff, punt, field goal, extra point, no-play, or
  any other non-scrimmage row is `0`, because the previous play type is not
  `run`, `pass`, or `qb_kneel`.
* The first play of a new quarter is `0`, because a quarter boundary ends the
  running-clock episode.
* Rows after an incomplete pass, an out-of-bounds finish, a two-minute-warning
  play, an end-of-quarter play, or a timeout are `0`.
* The proxy approximates clock status and does not reconstruct the actual game
  clock. Where the evidence is ambiguous it returns `0` rather than inventing
  a stopped-clock signal.

Missing raw flags are treated as `0` when evaluating the proxy, and raw flag
values are parsed numerically so that numeric, boolean, and numeric-string
encodings behave identically.

## Rejected information

The raw table is much wider than the 23 columns that are read. Everything
outside the accepted vocabulary is rejected because it either describes the
play being predicted or is computed from its result. The contract rejects
these families:

| Rejected family | Examples | Why |
| --- | --- | --- |
| Current-play outcome | `yards_gained` (current), `ydsnet`, `air_yards`, `yards_after_catch` | These are recorded after the snap, and `ydsnet` is drive net yards through the current play. |
| Current/cumulative EPA and WPA | `epa`, `wpa`, `air_epa`, `yac_epa`, `comp_air_epa`, `comp_yac_epa`, `total_home_epa`, `total_away_epa`, `total_home_rush_epa`, ..., all `*_wpa` counterparts | These are either the current play's result, such as `epa` and `wpa`, or a cumulative total that includes it. The contract rejects all current and cumulative EPA/WPA columns. |
| Expected points / win probability family | `ep`, `wp`, `def_wp`, `home_wp`, `away_wp`, `*_wp_post` | These are external model outputs outside the raw-context vocabulary. The `*_post` columns also leak the outcome. |
| Current play description | `desc` (current row) | This is free text written after the play, including the result. `desc` is read only to evaluate the *previous* row for the clock proxy. |
| Current formation/pace | `shotgun`, `no_huddle`, `qb_dropback`, `qb_scramble`, `pass_length`, `pass_location`, `run_location`, `run_gap` | These describe the current play itself, and conditioning on them would be circular for a pre-call model. |
| Outcome flags | `first_down_*`, `third_down_*`, `fourth_down_*`, `incomplete_pass`, `interception`, `sack`, `fumble_*`, `touchdown`, `pass_touchdown`, `rush_touchdown`, `return_touchdown`, `punt_*`, `kickoff_*`, `field_goal_result`, `kick_distance`, `extra_point_result`, `two_point_conv_result` | Binary post-play results. |
| Post-play score state | `posteam_score_post`, `defteam_score_post`, `score_differential_post` | Updated after the play. |
| Current-play penalties | `penalty_team`, `penalty_type`, `penalty_yards` | These are consequences of the current play. Only the previous row's `penalty` flag is used, and only for the clock proxy. |
| Administration and identity | `drive`, `sp`, `time`, `yrdln`, `side_of_field`, `game_seconds_remaining`, `game_half`, `goal_to_go`, player ids/names, `timeout_team`, `replay_or_challenge`, ... | These fall outside the raw-context vocabulary. Player identity in particular invites memorization rather than learning from context. |

`yards_gained`, `desc`, `qb_spike`, `quarter_end`, `penalty`, and `timeout`
are among the read columns, but only their shifted previous-row values
influence the output, and none of them leave `prepare_plays` as raw columns.

## Missing values and the imputation boundary

`prepare_plays` never fills values. Numeric gaps such as `down` on a two-point
attempt stay `NaN`, and categorical gaps stay missing as `NaN` in an
`object`-dtype column, which is the representation scikit-learn imputers
expect. Imputation belongs to the model pipeline and must be fitted on the
training partition only, using a median numeric imputer and a most-frequent
categorical imputer, never on the holdout. This boundary fixes the leakage in
the 2024 experiment, where scaling was fitted on the full dataset before
splitting.

## `chronological_split`

```python
train, test = chronological_split(prepared, holdout_start="2017-09-01")
```

* Training rows have a `game_date` strictly before the cutoff, and holdout
  rows have one on or after it. With the default cutoff, every game before
  September 2017 lands in train, and every game from September 2017 onward
  forms the future-season holdout, which covers the 2017 and 2018 seasons.
* Every game is kept whole. The function rejects any split where a `game_id`
  would appear in both partitions, which would also break the
  one-date-per-game invariant enforced earlier.
* Empty partitions, missing identifiers, invalid dates, and an invalid cutoff
  raise `ValueError`, and a non-DataFrame argument raises `TypeError`.
* Callers such as the tests or the smoke fixture may pass an earlier cutoff;
  the function is not hard-coded to the default.
* Both returned frames have a fresh index.

A random row-level split would let plays from the same game and the same
season appear on both sides, which makes the score optimistic and says nothing
about future seasons. The chronological, game-separated holdout measures what
the study claims to measure: generalization to games played later.

## Output schema of `prepare_plays`

| Position | Column | Type |
| --- | --- | --- |
| 1 | `game_id` | `int64`, or a trimmed string for non-numeric identifiers |
| 2 | `play_id` | `int64`, or a trimmed string for non-numeric identifiers |
| 3 | `game_date` | `datetime64[ns]`, midnight-normalized |
| 4 | `play_type` | the target label, one of `TARGET_LABELS` |
| 5-15 | `NUMERIC_FEATURES` in contract order | numeric; `previous_yards_gained` is `float64` and holds `NaN` on game openers, and `clock_running_proxy` is `int8` holding 0 or 1 |
| 16-20 | `CATEGORICAL_FEATURES` in contract order | `object` text with missing values preserved as `NaN` |

Rows are ordered by `(game_id, play_id)` and the index is a fresh
`RangeIndex`.

## Usage

```python
import pandas as pd

from nfl_showcase.features import (
    RAW_COLUMNS,
    chronological_split,
    clean_raw_plays,
    prepare_plays,
)

raw = pd.read_csv(
    "NFL Play by Play 2009-2018 (v5).csv",
    usecols=RAW_COLUMNS,
    encoding="utf-8-sig",
)
# This source has 2,393 exact repeats and one ambiguous game, so handle both
# before the strict preparation step; see data.md for the counts.
cleaned, summary = clean_raw_plays(raw, exclude_ambiguous_games=True)
prepared = prepare_plays(cleaned)
train, test = chronological_split(prepared, holdout_start="2017-09-01")
print(summary)
print(len(train), len(test), sorted(prepared["play_type"].unique()))
```

## Known limitations

* `clock_running_proxy` approximates the clock using only the previous row. It
  is not ground truth, and it returns `0` when the evidence is unclear.
* The real source contains repeated records and one ambiguous game, and
  `clean_raw_plays` is the only place that handles them. Exact repeats
  collapse to the first record, and the ambiguous game is dropped only when
  the caller opts in with `exclude_ambiguous_games=True`. The default rejects
  the source, and there is no keep-last, averaging, or fill rule.
* `previous_play_type` keeps raw special-teams and no-play labels such as
  `kickoff`, `extra_point`, and `no_play` alongside the six target labels and
  `unknown`. The one-hot encoder ignores unseen values at inference time.
* Team normalization collapses franchise history, so Raiders rows from Oakland
  and Las Vegas share `LV`. That is intentional for a single categorical
  identity, but it discards era information.
* The label distribution is imbalanced, and `qb_spike` and `qb_kneel` are
  rare. The reported metrics are macro-F1 and per-class support alongside
  accuracy, not accuracy alone.
* Feature preparation does not check whether numeric feature values are
  plausible, so a row with a negative `ydstogo` would flow into the
  training-only imputers as it is. The historical source data contains no such
  rows.
