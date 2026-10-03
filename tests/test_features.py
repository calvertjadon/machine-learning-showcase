"""Consumer-visible behavior of the public features interface."""

from __future__ import annotations

import pandas as pd
import pytest

from helpers import make_plays
from nfl_showcase.features import (
    FEATURE_COLUMNS,
    TARGET_LABELS,
    chronological_split,
    clean_raw_plays,
    prepare_plays,
)


def _prepare_sorted(raw: pd.DataFrame) -> pd.DataFrame:
    return prepare_plays(raw).sort_values(["game_id", "play_id"]).reset_index(drop=True)


def _multi_date_frame() -> pd.DataFrame:
    return make_plays(
        {"game_id": 1, "game_date": "2016-09-11", "play_type": "pass"},
        {"game_id": 1, "game_date": "2016-09-11", "play_type": "run", "yards_gained": 3.0},
        {"game_id": 2, "game_date": "2016-10-02", "play_type": "pass", "yards_gained": 5.0},
        {"game_id": 3, "game_date": "2017-09-10", "play_type": "run", "yards_gained": 2.0},
        {"game_id": 3, "game_date": "2017-09-10", "play_type": "pass", "yards_gained": 6.0},
    )


def test_prepare_plays_emits_context_features_for_target_plays_only():
    raw = make_plays(
        {"play_type": "pass", "yards_gained": 9.0},
        {"play_type": "run", "yards_gained": 3.0},
        {"play_type": "kickoff", "down": None, "ydstogo": None, "yards_gained": 60.0},
        {"play_type": "punt", "down": 4, "ydstogo": 12, "yards_gained": 42.0},
    )

    prepared = prepare_plays(raw)

    assert list(prepared.columns) == [
        "game_id",
        "play_id",
        "game_date",
        "play_type",
        *FEATURE_COLUMNS,
    ]
    assert set(prepared["play_type"]) == {"pass", "run", "punt"}
    assert "desc" not in prepared.columns
    assert "yards_gained" not in prepared.columns
    assert len(prepared) == 3


def test_previous_context_is_shifted_within_each_game_after_interleaving():
    raw = make_plays(
        {"game_id": 1, "play_id": 1, "yards_gained": 7.0},
        {
            "game_id": 2,
            "play_id": 1,
            "home_team": "KC",
            "away_team": "DEN",
            "posteam": "KC",
            "defteam": "DEN",
            "yards_gained": 41.0,
        },
        {"game_id": 1, "play_id": 2, "yards_gained": 3.0},
        {
            "game_id": 2,
            "play_id": 2,
            "home_team": "KC",
            "away_team": "DEN",
            "posteam": "KC",
            "defteam": "DEN",
            "yards_gained": -2.0,
        },
    )

    prepared = _prepare_sorted(raw)
    first_game = prepared[prepared["game_id"] == 1].reset_index(drop=True)
    second_game = prepared[prepared["game_id"] == 2].reset_index(drop=True)

    assert first_game.loc[0, "previous_play_type"] == "unknown"
    assert pd.isna(first_game.loc[0, "previous_yards_gained"])
    assert first_game.loc[1, "previous_yards_gained"] == 7.0
    assert first_game.loc[1, "previous_play_type"] == "pass"
    assert second_game.loc[1, "previous_yards_gained"] == 41.0
    assert second_game.loc[1, "previous_play_type"] == "pass"


def test_first_play_of_a_game_resets_previous_context():
    prepared = _prepare_sorted(make_plays({"yards_gained": 11.0}, {"yards_gained": 4.0}))

    first, second = prepared.iloc[0], prepared.iloc[1]

    assert first["previous_play_type"] == "unknown"
    assert pd.isna(first["previous_yards_gained"])
    assert not bool(first["clock_running_proxy"])
    assert second["previous_play_type"] == "pass"
    assert second["previous_yards_gained"] == 11.0
    assert bool(second["clock_running_proxy"])


def test_clock_running_proxy_resets_at_quarter_boundaries():
    raw = make_plays(
        {"qtr": 1, "quarter_seconds_remaining": 12.0, "yards_gained": 6.0},
        {"qtr": 1, "quarter_seconds_remaining": 6.0, "play_type": "run", "yards_gained": 2.0},
        {"qtr": 2, "quarter_seconds_remaining": 880.0, "yards_gained": 3.0},
    )

    prepared = _prepare_sorted(raw)

    assert bool(prepared.loc[1, "clock_running_proxy"])
    assert not bool(prepared.loc[2, "clock_running_proxy"])
    assert prepared.loc[2, "previous_play_type"] == "run"
    assert prepared.loc[2, "previous_yards_gained"] == 2.0


@pytest.mark.parametrize(
    "stopped_prior",
    [
        {"desc": "pass incomplete"},
        {"desc": "run out of bounds"},
        {"desc": "two-minute warning"},
        {"desc": "end of quarter"},
        {"play_type": "run", "desc": "timeout called"},
        {"play_type": "qb_spike", "qb_spike": 1, "desc": "spiked the ball"},
        {"play_type": "run", "quarter_end": 1, "desc": "run up the middle"},
        {"play_type": "run", "penalty": 1, "desc": "run up the middle"},
        {"play_type": "run", "timeout": 1, "desc": "timeout"},
        {"play_type": "kickoff", "down": None, "ydstogo": None, "desc": "kickoff 60 yards"},
        {"play_type": "no_play", "down": None, "ydstogo": None, "desc": "no play"},
    ],
)
def test_clock_running_proxy_stays_zero_after_a_stopped_clock(stopped_prior):
    raw = make_plays(stopped_prior, {"play_type": "pass", "yards_gained": 5.0})

    prepared = _prepare_sorted(raw)

    target_proxy = prepared.loc[prepared["play_id"] == 2, "clock_running_proxy"]
    assert target_proxy.tolist() == [0]
    # Non-target predecessors such as kickoff and no_play must not survive
    # preparation.
    assert set(prepared["play_type"]) <= set(TARGET_LABELS)


def test_context_is_derived_before_filtering_to_target_labels():
    filtered_before = make_plays(
        {
            "play_type": "kickoff",
            "down": None,
            "ydstogo": None,
            "yards_gained": 55.0,
            "desc": "kickoff 55 yards",
        },
        {"play_type": "pass", "yards_gained": 6.0},
    )

    prepared = _prepare_sorted(filtered_before)

    assert len(prepared) == 1
    assert prepared.loc[0, "previous_play_type"] == "kickoff"
    assert prepared.loc[0, "previous_yards_gained"] == 55.0
    assert not bool(prepared.loc[0, "clock_running_proxy"])

    filtered_between = make_plays(
        {"play_type": "run", "yards_gained": 2.0},
        {"play_type": "extra_point", "down": None, "ydstogo": None, "yards_gained": 0.0},
        {"play_type": "pass", "yards_gained": 8.0},
    )

    prepared = _prepare_sorted(filtered_between)

    assert list(prepared["play_id"]) == [1, 3]
    assert set(prepared["play_type"]) == {"run", "pass"}
    assert prepared.loc[1, "previous_play_type"] == "extra_point"
    assert prepared.loc[1, "previous_yards_gained"] == 0.0


@pytest.mark.parametrize(
    ("raw_team", "normalized"),
    [("OAK", "LV"), ("SD", "LAC"), ("STL", "LAR"), ("LA", "LAR"), ("JAC", "JAX")],
)
def test_relocated_team_codes_are_normalized(raw_team, normalized):
    raw = make_plays(
        {
            "home_team": raw_team,
            "away_team": "BUF",
            "posteam": raw_team,
            "defteam": "BUF",
        }
    )

    prepared = prepare_plays(raw)

    assert prepared.loc[0, "home_team"] == normalized
    assert prepared.loc[0, "posteam"] == normalized
    assert prepared.loc[0, "away_team"] == "BUF"
    assert prepared.loc[0, "defteam"] == "BUF"


def test_missing_previous_values_are_preserved_for_training_only_imputation():
    raw = make_plays(
        {"play_type": "pass", "yards_gained": float("nan"), "desc": "pass incomplete"},
        {"play_type": "run", "yards_gained": 4.0},
    )

    prepared = _prepare_sorted(raw)

    assert pd.isna(prepared.loc[0, "previous_yards_gained"])
    assert pd.isna(prepared.loc[1, "previous_yards_gained"])
    assert prepared.loc[1, "previous_play_type"] == "pass"


def test_blank_previous_play_label_maps_to_unknown():
    raw = make_plays(
        {"play_type": None, "down": None, "ydstogo": None},
        {"play_type": "run", "yards_gained": 4.0},
    )

    prepared = _prepare_sorted(raw)

    assert len(prepared) == 1
    assert prepared.loc[0, "previous_play_type"] == "unknown"


def test_prepare_plays_requires_a_dataframe():
    with pytest.raises(TypeError):
        prepare_plays([{"play_type": "pass"}])


def test_prepare_plays_rejects_missing_raw_columns():
    raw = make_plays({"yards_gained": 5.0}).drop(columns=["half_seconds_remaining"])

    with pytest.raises(ValueError):
        prepare_plays(raw)


def test_prepare_plays_rejects_duplicate_play_identifiers():
    raw = make_plays({"play_id": 1}, {"play_id": 1})

    with pytest.raises(ValueError):
        prepare_plays(raw)


def test_prepare_plays_rejects_conflicting_dates_for_one_game():
    raw = make_plays({"game_date": "2016-09-11"}, {"game_date": "2016-09-18"})

    with pytest.raises(ValueError):
        prepare_plays(raw)


def test_prepare_plays_rejects_unparseable_dates():
    raw = make_plays({"game_date": "definitely-not-a-date"})

    with pytest.raises(ValueError):
        prepare_plays(raw)


@pytest.mark.parametrize("column", ["game_id", "play_id"])
def test_prepare_plays_rejects_blank_identifiers(column):
    raw = make_plays({column: None})

    with pytest.raises(ValueError):
        prepare_plays(raw)


def test_prepare_plays_rejects_empty_frames():
    raw = make_plays().iloc[0:0]

    with pytest.raises(ValueError):
        prepare_plays(raw)


def test_prepare_plays_rejects_frames_without_target_plays():
    raw = make_plays(
        {"play_type": "kickoff", "down": None, "ydstogo": None},
        {"play_type": "no_play", "down": None, "ydstogo": None},
    )

    with pytest.raises(ValueError):
        prepare_plays(raw)


def test_chronological_split_is_temporal_and_keeps_games_whole():
    prepared = prepare_plays(_multi_date_frame())

    train, test = chronological_split(prepared, holdout_start="2017-01-01")

    assert set(train["game_id"]) == {1, 2}
    assert set(test["game_id"]) == {3}
    assert set(train["game_id"]).isdisjoint(set(test["game_id"]))
    assert pd.to_datetime(train["game_date"]).max() < pd.Timestamp("2017-01-01")
    assert pd.to_datetime(test["game_date"]).min() >= pd.Timestamp("2017-01-01")
    assert len(train) + len(test) == len(prepared)


def test_chronological_split_default_holdout_starts_with_the_2017_season():
    prepared = prepare_plays(_multi_date_frame())

    train, test = chronological_split(prepared)

    assert pd.to_datetime(train["game_date"]).max() < pd.Timestamp("2017-09-01")
    assert pd.to_datetime(test["game_date"]).min() >= pd.Timestamp("2017-09-01")


def test_chronological_split_rejects_unparseable_cutoff():
    prepared = prepare_plays(_multi_date_frame())

    with pytest.raises(ValueError):
        chronological_split(prepared, holdout_start="not-a-date")


@pytest.mark.parametrize("cutoff", ["2016-01-01", "2018-01-01"])
def test_chronological_split_rejects_empty_sides(cutoff):
    prepared = prepare_plays(_multi_date_frame())

    with pytest.raises(ValueError):
        chronological_split(prepared, holdout_start=cutoff)


def test_chronological_split_rejects_games_that_straddle_the_cutoff():
    prepared = prepare_plays(_multi_date_frame())
    straddling = prepared.copy()
    game_one_rows = straddling.index[straddling["game_id"] == 1]
    straddling.loc[game_one_rows[1], "game_date"] = pd.Timestamp("2017-06-01")

    with pytest.raises(ValueError):
        chronological_split(straddling, holdout_start="2017-01-01")


def test_chronological_split_rejects_empty_prepared_frames():
    prepared = prepare_plays(_multi_date_frame()).iloc[0:0]

    with pytest.raises(ValueError):
        chronological_split(prepared, holdout_start="2017-01-01")


def test_clean_raw_plays_collapses_exact_repeats_without_a_self_predecessor():
    raw = make_plays(
        {
            "game_id": 2,
            "play_id": 2,
            "game_date": "2016-10-02",
            "home_team": "KC",
            "away_team": "DEN",
            "posteam": "KC",
            "defteam": "DEN",
            "play_type": "run",
            "yards_gained": 6.0,
        },
        {"game_id": 1, "play_id": 1, "play_type": "run", "yards_gained": 3.0},
        {
            "game_id": 2,
            "play_id": 1,
            "game_date": "2016-10-02",
            "home_team": "KC",
            "away_team": "DEN",
            "posteam": "KC",
            "defteam": "DEN",
            "play_type": "pass",
            "yards_gained": 9.0,
        },
        {"game_id": 1, "play_id": 2, "play_type": "pass", "yards_gained": 7.0},
        {"game_id": 1, "play_id": 1, "play_type": "run", "yards_gained": 3.0},
        {
            "game_id": 2,
            "play_id": 2,
            "game_date": "2016-10-02",
            "home_team": "KC",
            "away_team": "DEN",
            "posteam": "KC",
            "defteam": "DEN",
            "play_type": "run",
            "yards_gained": 6.0,
        },
        {"game_id": 1, "play_id": 3, "play_type": "run", "yards_gained": -1.0},
    )

    # Strict preparation still rejects the uncleaned source, because the
    # duplicated (game_id, play_id) pairs are an error.
    with pytest.raises(ValueError):
        prepare_plays(raw)

    # Cleaning collapses the non-adjacent whole-row repeats without any flags,
    # because the duplicate rows do not contradict each other.
    cleaned, summary = clean_raw_plays(raw)

    assert summary["raw_rows"] == 7
    assert summary["exact_duplicate_rows_removed"] == 2
    assert summary["ambiguous_id_groups"] == 0
    assert summary["ambiguous_games_excluded"] == 0
    assert summary["ambiguous_rows_removed"] == 0
    assert summary["retained_rows"] == 5
    assert len(cleaned) == 5

    prepared = prepare_plays(cleaned)
    assert len(prepared) == 5
    games = {
        int(game_id): rows.sort_values("play_id").reset_index(drop=True)
        for game_id, rows in prepared.groupby("game_id")
    }

    game_one = games[1]
    game_two = games[2]
    assert list(game_one["play_id"]) == [1, 2, 3]
    assert list(game_two["play_id"]) == [1, 2]
    # Each surviving play must see the preceding surviving play as its
    # previous event, never a removed copy of itself.
    assert game_one.loc[1, "previous_play_type"] == "run"
    assert game_one.loc[1, "previous_yards_gained"] == 3.0
    assert game_one.loc[2, "previous_play_type"] == "pass"
    assert game_one.loc[2, "previous_yards_gained"] == 7.0
    assert game_two.loc[1, "previous_play_type"] == "pass"
    assert game_two.loc[1, "previous_yards_gained"] == 9.0


@pytest.mark.parametrize("column", ["posteam_timeouts_remaining", "defteam_timeouts_remaining"])
def test_clean_raw_plays_rejects_same_ids_that_differ_only_in_timeouts(column):
    raw = make_plays(
        {"game_id": 1, "play_id": 1, "play_type": "pass", "yards_gained": 5.0},
        {"game_id": 1, "play_id": 1, "play_type": "pass", "yards_gained": 5.0, column: 1},
        {"game_id": 1, "play_id": 2, "play_type": "run", "yards_gained": 3.0},
    )

    with pytest.raises(ValueError):
        clean_raw_plays(raw)


def test_clean_raw_plays_opt_in_excludes_whole_ambiguous_games_and_keeps_the_others():
    raw = make_plays(
        {"game_id": 10, "play_id": 1, "play_type": "run", "yards_gained": 3.0},
        {"game_id": 10, "play_id": 2, "play_type": "pass", "yards_gained": 7.0},
        {"game_id": 10, "play_id": 2, "play_type": "pass", "yards_gained": 7.0},
        {"game_id": 20, "play_id": 1, "play_type": "run", "yards_gained": 2.0},
        {"game_id": 20, "play_id": 2, "play_type": "pass", "yards_gained": 9.0},
        {
            "game_id": 20,
            "play_id": 2,
            "play_type": "pass",
            "yards_gained": 9.0,
            "posteam_timeouts_remaining": 1,
        },
        {"game_id": 20, "play_id": 3, "play_type": "punt", "yards_gained": 40.0},
        {"game_id": 30, "play_id": 1, "play_type": "run", "yards_gained": 5.0},
        {"game_id": 30, "play_id": 1, "play_type": "run", "yards_gained": 5.0},
    )

    with pytest.raises(ValueError):
        clean_raw_plays(raw)

    cleaned, summary = clean_raw_plays(raw, exclude_ambiguous_games=True)

    assert summary["raw_rows"] == 9
    assert summary["exact_duplicate_rows_removed"] == 2
    assert summary["ambiguous_id_groups"] == 1
    assert summary["ambiguous_games_excluded"] == 1
    assert summary["ambiguous_rows_removed"] == 4
    assert summary["retained_rows"] == 3
    assert set(cleaned["game_id"]) == {10, 30}
    assert (cleaned["game_id"] == 10).sum() == 2
    assert (cleaned["game_id"] == 30).sum() == 1

    prepared = prepare_plays(cleaned)
    assert set(prepared["game_id"]) == {10, 30}
    game_ten = prepared[prepared["game_id"] == 10].sort_values("play_id").reset_index(drop=True)
    assert game_ten.loc[1, "previous_play_type"] == "run"
    assert game_ten.loc[1, "previous_yards_gained"] == 3.0


def test_clean_raw_plays_rejects_empty_input_and_an_empty_result():
    with pytest.raises(ValueError):
        clean_raw_plays(make_plays().iloc[0:0])

    only_ambiguous_game = make_plays(
        {"game_id": 1, "play_id": 1, "play_type": "pass"},
        {"game_id": 1, "play_id": 1, "play_type": "pass", "posteam_timeouts_remaining": 1},
    )

    with pytest.raises(ValueError):
        clean_raw_plays(only_ambiguous_game, exclude_ambiguous_games=True)


@pytest.mark.parametrize(
    "bad_row",
    [
        {"game_date": "definitely-not-a-date"},
        {"game_id": None},
        {"play_id": None},
    ],
)
def test_clean_raw_plays_rejects_invalid_dates_and_identifiers(bad_row):
    raw = make_plays({"play_type": "pass"}, {"play_type": "run", **bad_row})

    with pytest.raises(ValueError):
        clean_raw_plays(raw)


def test_clean_raw_plays_rejects_conflicting_dates_for_one_game():
    raw = make_plays(
        {"game_id": 1, "play_id": 1, "play_type": "pass", "game_date": "2016-09-11"},
        {"game_id": 1, "play_id": 2, "play_type": "run", "game_date": "2016-09-18"},
    )

    with pytest.raises(ValueError):
        clean_raw_plays(raw)


@pytest.mark.parametrize(
    "broken_row",
    [
        {"game_id": 2, "play_id": 2, "play_type": "run", "game_date": "definitely-not-a-date"},
        {"game_id": 2, "play_id": None, "play_type": "run"},
    ],
)
def test_clean_raw_plays_does_not_hide_invalid_rows_in_an_excluded_game(broken_row):
    raw = make_plays(
        {"game_id": 1, "play_id": 1, "play_type": "pass"},
        {"game_id": 2, "play_id": 1, "play_type": "pass"},
        {"game_id": 2, "play_id": 1, "play_type": "pass", "posteam_timeouts_remaining": 1},
        broken_row,
    )

    with pytest.raises(ValueError):
        clean_raw_plays(raw, exclude_ambiguous_games=True)


def test_clean_prepare_and_split_stay_whole_game_and_causal_after_exclusion():
    raw = make_plays(
        {"game_id": 1, "play_id": 1, "play_type": "pass", "yards_gained": 4.0},
        {"game_id": 1, "play_id": 2, "play_type": "run", "yards_gained": 3.0},
        {"game_id": 1, "play_id": 1, "play_type": "pass", "yards_gained": 4.0},
        {
            "game_id": 2,
            "play_id": 1,
            "play_type": "pass",
            "yards_gained": 8.0,
            "game_date": "2017-09-10",
        },
        {
            "game_id": 2,
            "play_id": 2,
            "play_type": "run",
            "yards_gained": 2.0,
            "game_date": "2017-09-10",
        },
        {
            "game_id": 2,
            "play_id": 2,
            "play_type": "run",
            "yards_gained": 2.0,
            "game_date": "2017-09-10",
            "posteam_timeouts_remaining": 1,
        },
        {
            "game_id": 2,
            "play_id": 3,
            "play_type": "punt",
            "yards_gained": 40.0,
            "game_date": "2017-09-10",
        },
        {
            "game_id": 3,
            "play_id": 1,
            "play_type": "pass",
            "yards_gained": 11.0,
            "game_date": "2017-10-01",
        },
        {
            "game_id": 3,
            "play_id": 2,
            "play_type": "run",
            "yards_gained": 5.0,
            "game_date": "2017-10-01",
        },
    )

    cleaned, summary = clean_raw_plays(raw, exclude_ambiguous_games=True)

    assert summary["raw_rows"] == 9
    assert summary["exact_duplicate_rows_removed"] == 1
    assert summary["ambiguous_id_groups"] == 1
    assert summary["ambiguous_games_excluded"] == 1
    assert summary["ambiguous_rows_removed"] == 4
    assert summary["retained_rows"] == 4

    prepared = prepare_plays(cleaned)
    assert set(prepared["game_id"]) == {1, 3}

    train, test = chronological_split(prepared, holdout_start="2017-01-01")

    assert set(train["game_id"]) == {1}
    assert set(test["game_id"]) == {3}
    assert len(train) + len(test) == len(prepared)

    game_one = train.sort_values("play_id").reset_index(drop=True)
    game_three = test.sort_values("play_id").reset_index(drop=True)
    assert list(game_one["play_id"]) == [1, 2]
    assert list(game_three["play_id"]) == [1, 2]
    assert game_one.loc[1, "previous_play_type"] == "pass"
    assert game_one.loc[1, "previous_yards_gained"] == 4.0
    assert game_three.loc[0, "previous_play_type"] == "unknown"
    assert pd.isna(game_three.loc[0, "previous_yards_gained"])
    assert game_three.loc[1, "previous_play_type"] == "pass"
    assert game_three.loc[1, "previous_yards_gained"] == 11.0
