"""Causal feature engineering for the NFL play-call showcase.

The raw play-by-play table mixes three kinds of columns: pre-snap game state,
descriptors of the current play, and post-play outcomes.  :func:`prepare_plays`
keeps only information that was honestly available *before* the snap of the
play whose type is being predicted, derives previous-play context inside each
game (never across game boundaries), and returns rows labelled with one of
:data:`TARGET_LABELS`.

:func:`clean_raw_plays` is the explicit, auditable step that handles the
repeated records and conflicting identifiers present in the published source:
exact repeats are collapsed, and games whose identifiers disagree are dropped
only when the caller opts in.  :func:`prepare_plays` keeps its strict
duplicate rejection unchanged and is meant to run on cleaned rows.

:func:`chronological_split` then carves a game-separated, future-season
holdout out of the prepared frame.

Modelling rules enforced here:

* every row is ordered by ``(game_id, play_id)`` before any ``shift`` and
  before any label filtering, so previous-play features are strictly causal;
* ``shift`` is computed per ``game_id``; the first play of a game carries no
  previous-play context;
* the previous play's outcome columns are allowed (``previous_yards_gained``),
  the current play's outcome columns are never used;
* missing values are preserved: numeric gaps stay ``NaN`` and categorical
  gaps stay missing so that imputers can be fitted on the training split only;
* only identifiers, the game date, the label, and :data:`FEATURE_COLUMNS`
  leave :func:`prepare_plays`.

The columns read from the raw file are exactly :data:`RAW_COLUMNS`; the raw
CSV is expected to be UTF-8 with a BOM (read it with ``encoding="utf-8-sig"``).
"""

import numpy as np
import pandas as pd

#: Minimal set of raw play-by-play columns required to prepare features.
#: The raw CSV ships with a UTF-8 BOM, so read it with ``utf-8-sig``.
RAW_COLUMNS: list[str] = [
    "game_id",
    "play_id",
    "game_date",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "yardline_100",
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "qtr",
    "down",
    "ydstogo",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "score_differential",
    "play_type",
    "yards_gained",
    "desc",
    "qb_spike",
    "quarter_end",
    "penalty",
    "timeout",
]

#: Numeric model inputs.  All are pre-snap except the two previous-play
#: features, which only depend on the preceding row of the same game.
NUMERIC_FEATURES: list[str] = [
    "yardline_100",
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "qtr",
    "down",
    "ydstogo",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "score_differential",
    "previous_yards_gained",
    "clock_running_proxy",
]

#: Categorical model inputs; missing values are kept for train-fitted
#: imputers, and relocated franchises are normalized to current names.
CATEGORICAL_FEATURES: list[str] = [
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "previous_play_type",
]

#: Complete model input vocabulary.
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Supported play-call labels, sorted; all other raw play types are treated
#: as context only and are filtered out after previous-play context exists.
TARGET_LABELS: list[str] = [
    "field_goal",
    "pass",
    "punt",
    "qb_kneel",
    "qb_spike",
    "run",
]

#: Identifier/label columns returned alongside the model features.
_CONTEXT_COLUMNS: tuple[str, ...] = ("game_id", "play_id", "game_date", "play_type")

#: Franchise moves between the 2009-2018 seasons, normalized to the current
#: abbreviation so that one franchise keeps a single categorical identity.
_TEAM_RELOCATIONS: dict[str, str] = {
    "OAK": "LV",  # Raiders, Las Vegas since 2020
    "SD": "LAC",  # Chargers, Los Angeles since 2017
    "STL": "LAR",  # Rams, Los Angeles since 2016
    "LA": "LAR",  # source abbreviation for the Rams after returning to LA
    "JAC": "JAX",  # source abbreviation for the Jaguars
}

#: A previous play only supports a running-clock claim if it was a normal
#: scrimmage play.
_CLOCK_RUNNING_PLAY_TYPES: frozenset[str] = frozenset({"run", "pass", "qb_kneel"})

#: Previous-play descriptions that imply the clock stopped.
_CLOCK_EXCLUSION_PATTERN: str = r"incomplete|out of bounds|two-minute|end of quarter|timeout"

#: Text values that mean "missing" when they appear in an identifier column.
_BLANK_TOKENS: frozenset[str] = frozenset({"na", "n/a", "nan", "none", "null"})


def _clean_text(series: pd.Series) -> pd.Series:
    """Strip surrounding whitespace and turn blank text into ``NaN``.

    The result is an ``object`` column where a gap is represented by
    ``numpy.nan``; this is the representation that scikit-learn's
    most-frequent imputer and one-hot encoder handle without extension-array
    quirks.  Pandas ``NA``/``NaT``/``None`` gaps are normalized to ``NaN``.
    """
    text = series.astype("object").where(series.notna(), np.nan)
    stripped = text.str.strip()
    return stripped.replace("", np.nan)


def _flag(series: pd.Series) -> pd.Series:
    """Interpret a raw 0/1 flag column as booleans, treating gaps as 0."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(0).ne(0)


def _normalize_identifiers(series: pd.Series, name: str) -> pd.Series:
    """Validate an identifier column and normalize whole numbers to ``int64``.

    Non-numeric identifiers are kept as trimmed strings; whole numeric
    identifiers are converted so that sorting and duplicate detection are
    numeric rather than lexicographic.
    """
    if series.isna().any():
        raise ValueError(f"{name} has missing values; every row needs a valid {name}")

    text = series.astype("string").str.strip()
    blank = text.eq("") | text.str.lower().isin(_BLANK_TOKENS)
    if blank.any():
        raise ValueError(f"{name} has blank values; every row needs a valid {name}")

    numeric = pd.to_numeric(text, errors="coerce")
    if numeric.notna().all():
        if not (numeric % 1 == 0).all():
            raise ValueError(f"{name} must contain whole numbers when numeric")
        return numeric.astype("int64")
    return text.astype(object)


def _normalize_dates(series: pd.Series) -> pd.Series:
    """Parse a date column, rejecting gaps, and normalize to midnight."""
    try:
        parsed = pd.to_datetime(series, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise ValueError("game_date values could not be parsed as dates") from exc
    if bool(parsed.isna().any()):
        raise ValueError("game_date has missing or invalid dates")
    return parsed.dt.normalize()


def _parse_cutoff(value: str | pd.Timestamp) -> pd.Timestamp:
    """Parse ``holdout_start`` into a midnight-normalized, naive timestamp."""
    try:
        cutoff = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"holdout_start is not a valid date: {value!r}") from exc
    if not isinstance(cutoff, pd.Timestamp) or pd.isna(cutoff):
        raise ValueError(f"holdout_start is not a valid date: {value!r}")
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    return cutoff.normalize()


def clean_raw_plays(
    raw: pd.DataFrame, *, exclude_ambiguous_games: bool = False
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean repeated and ambiguous raw rows before :func:`prepare_plays`.

    The published source file contains a small number of repeated records and a
    few ``(game_id, play_id)`` identifiers whose selected context disagrees
    (the same play number described with different pre-snap values inside one
    game).  :func:`prepare_plays` deliberately rejects any shared identifier,
    so a caller handling the real source must decide explicitly how to treat
    those rows first; this function is that explicit, auditable step.

    Cleaning is defined only in the selected :data:`RAW_COLUMNS` schema:

    1. identifiers and dates are normalized and validated the same way
       :func:`prepare_plays` validates them (missing columns, blank
       identifiers, unparseable dates, and games with conflicting dates raise
       ``ValueError``);
    2. records that are identical across every selected column are repeated
       rows; all but the first occurrence are dropped;
    3. afterwards, any ``(game_id, play_id)`` shared by more than one row is
       *ambiguous*: one identifier is described by different selected values
       and the source alone cannot say which record is the pre-snap truth.  No
       keep-last, ordering or averaging rule is invented.

    Args:
        raw: Raw play-by-play frame containing at least :data:`RAW_COLUMNS`.
            Extra columns are ignored and are not part of the returned frame.
        exclude_ambiguous_games: When ``False`` (default) ambiguous identifiers
            raise ``ValueError``.  When ``True``, every row of every affected
            game is dropped -- the whole game, not just the conflicting plays
            -- so previous-play context and whole-game evaluation never cross
            a partially removed game.

    Returns:
        ``(cleaned, summary)``.  ``cleaned`` contains exactly
        :data:`RAW_COLUMNS`, keeps the original row order (first occurrence
        first) and has a fresh ``RangeIndex``.  ``summary`` reports plain
        Python ``int`` counts:

        * ``raw_rows`` -- rows in ``raw`` before any cleaning;
        * ``exact_duplicate_rows_removed`` -- repeated records dropped;
        * ``ambiguous_id_groups`` -- ``(game_id, play_id)`` groups that still
          disagree after duplicate removal (``0`` means the source is
          unambiguous);
        * ``ambiguous_games_excluded`` -- distinct games dropped because the
          caller opted in (``0`` when none);
        * ``ambiguous_rows_removed`` -- rows dropped by that whole-game
          exclusion, counted after duplicate removal;
        * ``retained_rows`` -- rows returned in ``cleaned``.

        Excluded game identities are not reported in the summary.

    Raises:
        TypeError: If ``raw`` is not a :class:`pandas.DataFrame`.
        ValueError: If required columns are missing, the frame is empty,
            identifiers are missing/blank, dates are missing or invalid, a
            game has conflicting dates, ambiguity remains while
            ``exclude_ambiguous_games`` is ``False``, or no rows remain.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"raw must be a pandas DataFrame, got {type(raw).__name__}")

    missing = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError("raw play data is missing required columns: " + ", ".join(missing))
    if raw.empty:
        raise ValueError("raw play data is empty")

    plays = raw.loc[:, RAW_COLUMNS].copy()
    plays["game_id"] = _normalize_identifiers(plays["game_id"], "game_id")
    plays["play_id"] = _normalize_identifiers(plays["play_id"], "play_id")
    plays["game_date"] = _normalize_dates(plays["game_date"])

    dates_per_game = plays.groupby("game_id", sort=False)["game_date"].nunique()
    inconsistent = dates_per_game[dates_per_game > 1].index.tolist()
    if inconsistent:
        sample = inconsistent[:3]
        raise ValueError(f"{len(inconsistent)} game(s) have more than one game_date: {sample}")

    raw_rows = int(len(plays))

    exact_repeats = plays.duplicated(subset=list(RAW_COLUMNS), keep="first")
    exact_duplicate_rows_removed = int(exact_repeats.sum())
    deduplicated = plays.loc[~exact_repeats].reset_index(drop=True)

    # After exact duplicate removal, two rows can only share an identifier if
    # they disagree on at least one selected value, which is exactly the
    # ambiguous case the contract refuses to resolve silently.
    identifier_sizes = deduplicated.groupby(["game_id", "play_id"], sort=False).size()
    ambiguous_groups = identifier_sizes[identifier_sizes > 1]
    ambiguous_id_groups = int(ambiguous_groups.size)
    affected_games = ambiguous_groups.index.get_level_values("game_id").unique()

    if ambiguous_id_groups and not exclude_ambiguous_games:
        raise ValueError(
            f"{ambiguous_id_groups} (game_id, play_id) group(s) in "
            f"{len(affected_games)} game(s) have conflicting selected context after exact "
            "duplicate removal; pass exclude_ambiguous_games=True to drop every row of "
            "those games, or correct the source"
        )

    if ambiguous_id_groups:
        excluded = deduplicated["game_id"].isin(affected_games)
        ambiguous_rows_removed = int(excluded.sum())
        ambiguous_games_excluded = len(affected_games)
        cleaned = deduplicated.loc[~excluded].reset_index(drop=True)
    else:
        ambiguous_rows_removed = 0
        ambiguous_games_excluded = 0
        cleaned = deduplicated

    if cleaned.empty:
        raise ValueError(
            "no raw rows remain after cleaning; the source is only repeated or ambiguous"
        )

    summary = {
        "raw_rows": raw_rows,
        "exact_duplicate_rows_removed": exact_duplicate_rows_removed,
        "ambiguous_id_groups": ambiguous_id_groups,
        "ambiguous_games_excluded": ambiguous_games_excluded,
        "ambiguous_rows_removed": ambiguous_rows_removed,
        "retained_rows": int(len(cleaned)),
    }
    return cleaned, summary


def prepare_plays(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the causal, pre-snap feature frame from raw play-by-play rows.

    ``(game_id, play_id)`` pairs must be unique; this strictness is unchanged
    by design.  The published source file repeats some records and contains a
    few conflicting identifiers, so callers pass it through
    :func:`clean_raw_plays` first and prepare the cleaned frame.

    Args:
        raw: Raw play-by-play frame containing at least every column in
            :data:`RAW_COLUMNS`.  Extra columns are ignored.

    Returns:
        A new frame with one row per supported play (``play_type`` in
        :data:`TARGET_LABELS`), ordered by ``(game_id, play_id)`` and
        containing exactly ``game_id``, ``play_id``, ``game_date``,
        ``play_type`` and :data:`FEATURE_COLUMNS`.

    Raises:
        TypeError: If ``raw`` is not a :class:`pandas.DataFrame`.
        ValueError: If required columns are missing, the frame is empty,
            identifiers are missing/blank, dates are missing or invalid,
            ``(game_id, play_id)`` pairs are duplicated, a game has more than
            one date, or no supported target rows remain.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"raw must be a pandas DataFrame, got {type(raw).__name__}")

    missing = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError("raw play data is missing required columns: " + ", ".join(missing))
    if raw.empty:
        raise ValueError("raw play data is empty")

    plays = raw.loc[:, RAW_COLUMNS].copy()

    plays["game_id"] = _normalize_identifiers(plays["game_id"], "game_id")
    plays["play_id"] = _normalize_identifiers(plays["play_id"], "play_id")
    plays["game_date"] = _normalize_dates(plays["game_date"])

    duplicated = plays.duplicated(subset=["game_id", "play_id"])
    duplicate_count = int(duplicated.sum())
    if duplicate_count:
        raise ValueError(f"raw play data has {duplicate_count} duplicate (game_id, play_id) rows")

    dates_per_game = plays.groupby("game_id", sort=False)["game_date"].nunique()
    inconsistent = dates_per_game[dates_per_game > 1].index.tolist()
    if inconsistent:
        sample = inconsistent[:3]
        raise ValueError(f"{len(inconsistent)} game(s) have more than one game_date: {sample}")

    # Normalize text categories: relocated franchises and blank play labels.
    plays["play_type"] = _clean_text(plays["play_type"])
    for column in ("home_team", "away_team", "posteam", "defteam"):
        plays[column] = _clean_text(plays[column]).replace(_TEAM_RELOCATIONS)

    # Order every original row before shifting or filtering, so that each
    # previous-play feature refers to the immediately preceding play of the
    # same game.
    plays = plays.sort_values(["game_id", "play_id"], kind="stable")
    plays = plays.reset_index(drop=True)

    grouped = plays.groupby("game_id", sort=False)
    previous_play_type = grouped["play_type"].shift(1)
    previous_yards_gained = pd.to_numeric(grouped["yards_gained"].shift(1), errors="coerce")
    previous_quarter = grouped["qtr"].shift(1)
    previous_description = grouped["desc"].shift(1).astype("object")

    previous_spike = _flag(grouped["qb_spike"].shift(1))
    previous_quarter_end = _flag(grouped["quarter_end"].shift(1))
    previous_penalty = _flag(grouped["penalty"].shift(1))
    previous_timeout = _flag(grouped["timeout"].shift(1))
    has_previous_row = grouped.cumcount() > 0

    # Conservative clock-running proxy: claim the clock kept running only when
    # the preceding play of the same quarter was an ordinary scrimmage play
    # and nothing in its row or description suggests the clock stopped.
    clock_kept_running = (
        has_previous_row
        & plays["qtr"].notna()
        & previous_quarter.eq(plays["qtr"])
        & previous_play_type.isin(_CLOCK_RUNNING_PLAY_TYPES)
        & ~previous_spike
        & ~previous_quarter_end
        & ~previous_penalty
        & ~previous_timeout
        & ~previous_description.str.contains(
            _CLOCK_EXCLUSION_PATTERN, case=False, regex=True, na=False
        )
    )

    plays["previous_play_type"] = previous_play_type.fillna("unknown")
    plays["previous_yards_gained"] = previous_yards_gained
    plays["clock_running_proxy"] = clock_kept_running.fillna(False).astype("int8")

    # Filter to target labels only after the previous-play context has been
    # derived from every original row (kickoffs, punts, no-plays, timeouts).
    output_columns = [*_CONTEXT_COLUMNS, *FEATURE_COLUMNS]
    prepared = plays.loc[plays["play_type"].isin(TARGET_LABELS), output_columns]
    if prepared.empty:
        raise ValueError("no rows with a supported target play_type remain after preparation")
    return prepared.reset_index(drop=True)


def chronological_split(
    prepared: pd.DataFrame, holdout_start: str | pd.Timestamp = "2017-09-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a prepared frame into past and future partitions by game date.

    Every game stays whole: the training partition contains only rows dated
    strictly before ``holdout_start`` and the holdout partition only rows dated
    on or after it.  The default cutoff yields future seasons as the holdout
    for the 2009-2018 dataset.

    Args:
        prepared: Frame produced by :func:`prepare_plays` (needs at least
            ``game_id`` and ``game_date``).
        holdout_start: Date-like cutoff; defaults to ``"2017-09-01"``.

    Returns:
        ``(train, test)`` frames with reset indexes.

    Raises:
        TypeError: If ``prepared`` is not a :class:`pandas.DataFrame`.
        ValueError: If required columns are missing, the frame is empty,
            dates or the cutoff are invalid, a partition would be empty, or a
            game would be split across both partitions.
    """
    if not isinstance(prepared, pd.DataFrame):
        raise TypeError(f"prepared must be a pandas DataFrame, got {type(prepared).__name__}")

    missing = [column for column in ("game_id", "game_date") if column not in prepared.columns]
    if missing:
        raise ValueError("prepared play data is missing required columns: " + ", ".join(missing))
    if prepared.empty:
        raise ValueError("prepared play data is empty")

    game_ids = prepared["game_id"]
    if bool(game_ids.isna().any()):
        raise ValueError("game_id has missing values; cannot keep games whole")

    dates = _normalize_dates(prepared["game_date"])
    cutoff = _parse_cutoff(holdout_start)

    train_mask = dates < cutoff
    test_mask = dates >= cutoff
    if not bool(train_mask.any()):
        raise ValueError(f"no games fall before the holdout start {cutoff.date().isoformat()}")
    if not bool(test_mask.any()):
        raise ValueError(f"no games fall on or after the holdout start {cutoff.date().isoformat()}")

    overlap = set(game_ids[train_mask]) & set(game_ids[test_mask])
    if overlap:
        message = f"split would break up {len(overlap)} game(s); keep games whole"
        raise ValueError(message)

    train = prepared.loc[train_mask].reset_index(drop=True)
    test = prepared.loc[test_mask].reset_index(drop=True)
    return train, test
