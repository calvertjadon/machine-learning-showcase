"""Fixture builders shared by the consumer-visible feature tests.

All rows are original fictional plays. Every call to ``make_plays`` starts from
the ``RAW_COLUMNS`` defaults, and each override dict changes only the fields a
test examines.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from nfl_showcase.features import RAW_COLUMNS

_BASE_PLAY: dict[str, Any] = {
    "game_id": 2016091101,
    "play_id": 1,
    "game_date": "2016-09-11",
    "home_team": "NE",
    "away_team": "BUF",
    "posteam": "NE",
    "defteam": "BUF",
    "yardline_100": 65.0,
    "quarter_seconds_remaining": 900.0,
    "half_seconds_remaining": 1800.0,
    "qtr": 1,
    "down": 1,
    "ydstogo": 10,
    "posteam_timeouts_remaining": 3,
    "defteam_timeouts_remaining": 3,
    "score_differential": 0,
    "play_type": "pass",
    "yards_gained": 5.0,
    "desc": "short pass complete",
    "qb_spike": 0,
    "quarter_end": 0,
    "penalty": 0,
    "timeout": 0,
}


def make_plays(*overrides: dict[str, Any]) -> pd.DataFrame:
    """Build a raw play frame; each dict overrides one play's default values."""
    rows: list[dict[str, Any]] = []
    for position, override in enumerate(overrides, start=1):
        row = dict(_BASE_PLAY)
        row["play_id"] = position
        row.update(override)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))
