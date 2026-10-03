# Feature engineering and the evaluation split

This document audits how `nfl_showcase.features` turns raw play-by-play rows
into a **causal, pre-snap** feature frame, and how
`chronological_split` carves a game-separated, future-season holdout out of
it. Every prediction input is listed here together with the timing argument
that justifies it; everything rejected is listed with the reason it is
rejected. The audit is the contract that the models and CLI rely on.

The raw data is the public *NFL Play by Play 2009-2018 (v5)* CSV (about 450k
plays, 255 columns, UTF-8 with a BOM). The source file is **not** bundled in
this repository; see `docs/data.md` for download and rights information. That
verified file also contains repeated records and one game whose play
identifiers are described by conflicting pre-snap values, so
`clean_raw_plays` is the explicit, auditable gate for those rows; the exact
diagnosis is recorded in
[data.md](data.md#source-data-quality-and-cleaning-verified-file).

## Public interface

| Name | Contents |
| --- | --- |
| `RAW_COLUMNS` | The 23 raw columns read from the CSV: identifiers, date, teams, field position, clock, down/distance, timeouts, score, play type, and the previous-play auditing columns (`yards_gained`, `desc`, `qb_spike`, `quarter_end`, `penalty`, `timeout`). |
| `NUMERIC_FEATURES` | `yardline_100`, `quarter_seconds_remaining`, `half_seconds_remaining`, `qtr`, `down`, `ydstogo`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`, `score_differential`, `previous_yards_gained`, `clock_running_proxy`. |
| `CATEGORICAL_FEATURES` | `home_team`, `away_team`, `posteam`, `defteam`, `previous_play_type`. |
| `FEATURE_COLUMNS` | `NUMERIC_FEATURES + CATEGORICAL_FEATURES` (the complete model input vocabulary, in this exact order). |
| `TARGET_LABELS` | `field_goal`, `pass`, `punt`, `qb_kneel`, `qb_spike`, `run` (sorted). |
| `prepare_plays(raw)` | Returns `game_id`, `play_id`, `game_date`, `play_type` plus `FEATURE_COLUMNS` only. |
| `clean_raw_plays(raw, *, exclude_ambiguous_games=False)` | Explicit pre-preparation cleaning: collapses rows repeated across the 23 selected `RAW_COLUMNS` and, when opted in, drops the whole games affected by conflicting identifiers. Returns `(cleaned, summary)` with audited counts. |
| `chronological_split(prepared, holdout_start="2017-09-01")` | Returns `(train, test)`; train dates strictly before the cutoff, test dates on/after it, no game in both. |

The raw CSV is expected to be read with `usecols=RAW_COLUMNS` and
`encoding="utf-8-sig"`. The BOM is why the encoding matters: without it the
first column would arrive as `﻿play_id` and `prepare_plays` would (correctly)
reject the frame for a missing required column.

## `clean_raw_plays`: explicit handling of repeated and ambiguous rows

The verified source is not a perfectly canonical table. The "Source data
quality and cleaning" section of
[data.md](data.md#source-data-quality-and-cleaning-verified-file) records the
exact read-only diagnosis of the file identified by the SHA-256 listed there:
2,437 repeated `(game_id, play_id)` rows, of which 2,393 are fully identical
across the 23 selected columns. Collapsing those leaves 446,978 rows and 22
identifiers whose remaining records still disagree; all 22 are in a single
game dated 2011-12-04, which has 268 rows after exact repeat collapse. In
those conflict groups only `posteam_timeouts_remaining`/`defteam_timeouts_remaining`
differ; the source does not record which value was true pre-snap, and row
order is not evidence.

`prepare_plays` rejects shared identifiers outright, so real-data callers run
this explicit step first:

```python
cleaned, summary = clean_raw_plays(raw, exclude_ambiguous_games=True)
prepared = prepare_plays(cleaned)
```

All cleaning semantics are defined only in the selected 23 `RAW_COLUMNS`:

1. **Validation.** Missing required columns, an empty frame, missing/blank
   identifiers, missing or unparseable dates, and a game with conflicting
   dates raise `ValueError` — invalid-source errors are never hidden.
2. **Exact repeat collapse.** A record repeated across every selected column
   is a repeated row; all but the first occurrence are dropped. No other
   equality is considered: equal identifiers alone never justify deleting a
   row, and there is no keep-last rule.
3. **Ambiguity detection.** After repeat collapse, a `(game_id, play_id)`
   shared by more than one row necessarily disagrees on at least one selected
   value. That is an ambiguity the source cannot resolve, so no ordering,
   averaging, or fill rule is invented.
4. **Default rejection.** With `exclude_ambiguous_games=False`, residual
   ambiguity raises an actionable `ValueError` that reports the number of
   conflicting identifier groups and affected games and names the opt-in
   flag; affected game identities are not echoed.
5. **Opt-in whole-game exclusion.** With `exclude_ambiguous_games=True`, every
   row of every affected game is dropped — the whole game, not just the
   conflicting plays — so previous-event context and whole-game evaluation
   cannot cross a partially removed game. Rows and dates of unaffected games
   are preserved untouched.

The returned frame contains exactly the 23 selected columns in the original
row order (first occurrence kept, fresh `RangeIndex`). The summary has exactly
these keys, each a plain Python `int`:

| Key | Meaning |
| --- | --- |
| `raw_rows` | Rows in `raw` before any cleaning. |
| `exact_duplicate_rows_removed` | Repeated records collapsed. |
| `ambiguous_id_groups` | `(game_id, play_id)` groups still disagreeing after collapse (`0` if unambiguous). |
| `ambiguous_games_excluded` | Distinct games dropped by the opt-in (`0` when none). |
| `ambiguous_rows_removed` | Rows dropped by that whole-game exclusion, counted *after* repeat collapse. |
| `retained_rows` | Rows returned in `cleaned`. |

Excluded game identities are not part of the summary, and a result emptied by
exclusion raises `ValueError`. For the verified source, the counts are
2,393 collapsed exact repeats and, when the caller opts in, one game and 268
rows excluded, leaving 446,710 retained raw rows; `docs/data.md` states the
same facts from the source side. Cleaning does not change the source SHA-256
(that hash is of the full downloaded file).

## `prepare_plays` pipeline

The steps are ordered so that no information can leak backwards in time.

1. **Strict column check.** Every `RAW_COLUMNS` name must be present; extra
   columns are ignored. Missing columns raise `ValueError`.
2. **Non-empty check.** An empty raw frame raises `ValueError`.
3. **Identifier validation.** `game_id` and `play_id` must be non-missing and
   non-blank (blank includes the `NA` tokens the source file uses for missing
   values). Whole numeric identifiers are normalized to `int64` so that
   ordering and duplicate detection are numeric; non-numeric identifiers are
   kept as trimmed strings; fractional numeric identifiers are rejected.
4. **Date validation.** `game_date` is parsed with `errors="coerce"`;
   missing or unparseable values raise `ValueError`, and valid values are
   normalized to midnight. Dates are compared at date granularity.
5. **Duplicate check.** Duplicate `(game_id, play_id)` pairs raise
   `ValueError`. This strictness is unchanged by design: the real source is
   cleaned explicitly with `clean_raw_plays` *before* preparation (see above)
   rather than resolved implicitly inside the pipeline.
6. **One date per game.** Any game whose rows disagree on `game_date` raises
   `ValueError`; a game is a single calendar event and must not span a split.
7. **Text normalization.** `play_type` and the four team columns are
   whitespace-trimmed; blank text becomes a missing value. Relocated
   franchises are mapped to their current abbreviation:
   `OAK→LV`, `SD→LAC`, `STL→LAR`, `LA→LAR`, `JAC→JAX`.
8. **Global ordering.** *All original rows* are stable-sorted by
   `(game_id, play_id)` before any `shift` and before any label filtering.
   This is what makes the previous-play features causal and reproducible.
9. **Per-game previous-play context.** Every derived column is a
   `groupby("game_id").shift(1)` of the ordered rows, so context never crosses
   a game boundary. The first play of each game has no previous row.
10. **Clock-running proxy.** Derived from the previous row only (see below).
11. **Label filtering.** Rows are filtered to `TARGET_LABELS` **after** the
    previous-play context has been derived from every original row, so the
    first target play after a kickoff, punt, penalty, or timeout still sees
    that context honestly.
12. **Projection.** Only `game_id`, `play_id`, `game_date`, `play_type` and
    `FEATURE_COLUMNS` are returned. Raw descriptions and outcome columns never
    leave the function.

## Feature audit: what each predictor is and when it is known

Timing is stated relative to the snap of the play whose `play_type` is being
predicted. "Pre-snap" means the value is part of the game state a coach could
observe before calling the play; "previous play" means it is computed from the
immediately preceding row of the *same game* after the global sort.

| Feature | Source | Timing | Notes |
| --- | --- | --- | --- |
| `yardline_100` | `yardline_100` | Pre-snap | Distance from the opponent's goal line in yards (0-100). |
| `quarter_seconds_remaining` | `quarter_seconds_remaining` | Pre-snap | Seconds left in the current quarter. |
| `half_seconds_remaining` | `half_seconds_remaining` | Pre-snap | Seconds left in the current half. |
| `qtr` | `qtr` | Pre-snap | Period (1-5; 5 = overtime). |
| `down` | `down` | Pre-snap | Current down; missing values are kept (e.g. two-point tries). |
| `ydstogo` | `ydstogo` | Pre-snap | Yards to the first down (or goal). |
| `posteam_timeouts_remaining` | `posteam_timeouts_remaining` | Pre-snap | Timeouts left for the possession team; missing values kept. |
| `defteam_timeouts_remaining` | `defteam_timeouts_remaining` | Pre-snap | Timeouts left for the defence; missing values kept. |
| `score_differential` | `score_differential` | Pre-snap | Possession-team score minus opponent score at the snap. |
| `previous_yards_gained` | `yards_gained` shifted within the game | Previous play | Outcome of the immediately preceding play of the same game (a kickoff or a missing value yields `NaN`). |
| `clock_running_proxy` | derived `int8` 0/1 | Previous play | Conservative estimate that the game clock kept running after the previous play; 0 on a game's first row. |
| `home_team` | `home_team` | Pre-snap | Normalized abbreviation. |
| `away_team` | `away_team` | Pre-snap | Normalized abbreviation. |
| `posteam` | `posteam` | Pre-snap | Possession team; normalized abbreviation. |
| `defteam` | `defteam` | Pre-snap | Defence team; normalized abbreviation. |
| `previous_play_type` | `play_type` shifted within the game | Previous play | Raw previous-play type (`kickoff`, `punt`, `no_play`, special teams, any target label, ...) or `unknown` when the previous play's type is missing/blank or the row is a game opener. |

Identifier columns `game_id`, `play_id` and `game_date` are returned for
partitioning and reporting only; they are not model inputs. `play_type` is the
target label.

## The clock-running proxy in detail

Possession changes, penalties, spikes, timeouts and boundary plays all stop
the clock, but the raw table does not record a trustworthy "clock was running"
bit, so the module derives an explicit proxy from the previous row. The proxy
is deliberately conservative: it claims `1` only when

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

and `0` otherwise. Consequences:

* A game's first row is always `0`.
* The first play after a kickoff, punt, field goal, extra point, no-play, or
  any other non-scrimmage row is `0` (the previous play type is not
  `run`/`pass`/`qb_kneel`).
* The first play of a new quarter is `0`, because a quarter boundary ends the
  running-clock episode.
* Rows after an incomplete pass, an out-of-bounds finish, a two-minute-warning
  play, an end-of-quarter play, or a timeout are `0`.
* The proxy is an approximation of clock status, **not** a reconstruction of
  the actual game clock; where the evidence is ambiguous it errs toward `0`
  rather than inventing a stopped-clock signal.

Missing raw flags are treated as `0` when evaluating the proxy, and raw flag
values are parsed numerically so that numeric, boolean, and numeric-string
encodings behave identically.

## Rejected information

The raw table is much wider than the 23 columns read. Everything outside the
accepted vocabulary is rejected because it either describes the play being
predicted or is computed from its result. The contract's explicit rejections:

| Rejected family | Examples | Why |
| --- | --- | --- |
| Current-play outcome | `yards_gained` (current), `ydsnet`, `air_yards`, `yards_after_catch` | Recorded after the snap; `ydsnet` is drive net yards through the current play. |
| Current/cumulative EPA and WPA | `epa`, `wpa`, `air_epa`, `yac_epa`, `comp_air_epa`, `comp_yac_epa`, `total_home_epa`, `total_away_epa`, `total_home_rush_epa`, ..., all `*_wpa` counterparts | Either the current play's result (`epa`, `wpa`) or a cumulative total that includes it; the contract rejects all current/cumulative EPA/WPA. |
| Expected points / win probability family | `ep`, `wp`, `def_wp`, `home_wp`, `away_wp`, `*_wp_post` | External model outputs outside the audited raw-context vocabulary; the `*_post` columns additionally leak the outcome. |
| Current play description | `desc` (current row) | Free text written after the play, including the result; `desc` is read only to evaluate the *previous* row for the clock proxy. |
| Current formation/pace | `shotgun`, `no_huddle`, `qb_dropback`, `qb_scramble`, `pass_length`, `pass_location`, `run_location`, `run_gap` | Descriptors of the current play itself; conditioning on them would be circular for an honest pre-call model. |
| Outcome flags | `first_down_*`, `third_down_*`, `fourth_down_*`, `incomplete_pass`, `interception`, `sack`, `fumble_*`, `touchdown`, `pass_touchdown`, `rush_touchdown`, `return_touchdown`, `punt_*`, `kickoff_*`, `field_goal_result`, `kick_distance`, `extra_point_result`, `two_point_conv_result` | Binary post-play results. |
| Post-play score state | `posteam_score_post`, `defteam_score_post`, `score_differential_post` | Updated after the play. |
| Current-play penalties | `penalty_team`, `penalty_type`, `penalty_yards` | Consequences of the current play; only the previous row's `penalty` flag is used, for the clock proxy. |
| Administration and identity | `drive`, `sp`, `time`, `yrdln`, `side_of_field`, `game_seconds_remaining`, `game_half`, `goal_to_go`, player ids/names, `timeout_team`, `replay_or_challenge`, ... | Outside the contract's raw-context vocabulary; player identity in particular invites memorization rather than context learning. |

Note that `yards_gained`, `desc`, `qb_spike`, `quarter_end`, `penalty` and
`timeout` **are** among the read columns, but only their shifted
(previous-row) values influence the output, and none of them leave
`prepare_plays` as raw columns.

## Missing values and the imputation boundary

`prepare_plays` never fills values. Numeric gaps such as `down` on a two-point
attempt stay `NaN`; categorical gaps stay missing (`NaN` in an `object`-dtype
column, the representation scikit-learn imputers expect). Imputation is the
model pipeline's job and must be fitted **on the training partition only**
(median numeric imputer, most-frequent categorical imputer), never on the
holdout. This is the boundary that fixes the 2024 experiment's leakage, where
scaling was fitted on the full dataset before splitting.

## `chronological_split`

```python
train, test = chronological_split(prepared, holdout_start="2017-09-01")
```

* Training rows have `game_date` strictly before the cutoff; holdout rows have
  `game_date` on or after it. With the default cutoff, every game before
  September 2017 lands in train and every game from September 2017 onward
  (the 2017 and 2018 seasons) forms the future-season holdout.
* Every game is kept whole. The function rejects any split where a `game_id`
  would appear in both partitions (which would also break the one-date-per-game
  invariant enforced earlier).
* Empty partitions, missing identifiers, invalid dates, and an invalid cutoff
  raise `ValueError`; a non-DataFrame argument raises `TypeError`.
* Callers (for example tests or the smoke fixture) may pass an earlier cutoff;
  the function is not hard-coded to the default.
* Both returned frames have a fresh index.

The rationale: a random row-level split lets plays from the same game and the
same season appear on both sides, so the score is optimistic and says nothing
about future seasons. A chronological, game-separated holdout measures what
the historical project actually claimed to measure: generalization to games
played later.

## Output schema of `prepare_plays`

| Position | Column | Type |
| --- | --- | --- |
| 1 | `game_id` | `int64` (or trimmed string for non-numeric ids) |
| 2 | `play_id` | `int64` (or trimmed string for non-numeric ids) |
| 3 | `game_date` | `datetime64[ns]`, midnight-normalized |
| 4 | `play_type` | target label, one of `TARGET_LABELS` |
| 5-15 | `NUMERIC_FEATURES` in contract order | numeric; `previous_yards_gained` is `float64` with `NaN` on game openers, `clock_running_proxy` is `int8` 0/1 |
| 16-20 | `CATEGORICAL_FEATURES` in contract order | `object` text, missing values preserved (`NaN`) |

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
# This source has 2,393 exact repeats and one ambiguous game; handle both
# explicitly before the strict preparation step (see data.md for the counts).
cleaned, summary = clean_raw_plays(raw, exclude_ambiguous_games=True)
prepared = prepare_plays(cleaned)
train, test = chronological_split(prepared, holdout_start="2017-09-01")
print(summary)
print(len(train), len(test), sorted(prepared["play_type"].unique()))
```

## Known limitations

* `clock_running_proxy` is an explicit approximation built only from the
  previous row; it is not ground truth and errs toward `0`.
* The real source contains repeated records and one ambiguous game. They are
  handled only by the explicit `clean_raw_plays` contract above: exact repeats
  collapse keeping the first record, and the ambiguous game is dropped only
  when the caller opts in with `exclude_ambiguous_games=True`. The default
  rejects the source, and there is no keep-last, averaging, or fill rule.
* `previous_play_type` preserves raw special-teams and no-play labels
  (`kickoff`, `extra_point`, `no_play`, ...) in addition to the six target
  labels and `unknown`; the one-hot encoder ignores unseen values at inference
  time.
* Team normalization collapses franchise history (for example Raiders rows
  from Oakland and Las Vegas share `LV`), which is intentional for a single
  categorical identity but discards era information.
* The label distribution is imbalanced (`qb_spike` and `qb_kneel` are rare);
  macro-F1 and per-class support, not accuracy alone, are the reported
  metrics.
* Feature preparation does not validate the plausibility of numeric feature
  values (for example a negative `ydstogo`); such rows would flow into the
  training-only imputers as-is. The historical source data does not contain
  them.
