"""Deterministic mechanics smoke run for the public pipeline.

The fixture in this module is original fictional play-by-play data authored for this
repository. It exercises the full public pipeline (prepare -> split -> fit -> save/load
-> evaluate) without any real NFL data. Metrics emitted here describe pipeline wiring
only; they are explicitly not performance claims about real data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from nfl_showcase.features import (
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    TARGET_LABELS,
    chronological_split,
    prepare_plays,
)
from nfl_showcase.models import evaluate_models, fit_models, load_bundle, save_bundle

SMOKE_SEED = 42
SMOKE_TREES = 8
SMOKE_JOBS = 1
SMOKE_HOLDOUT_START = "2017-09-01"
SMOKE_BUNDLE_NAME = "smoke_models.joblib"
SMOKE_REPORT_NAME = "smoke_report.json"


class _Play(NamedTuple):
    """One scripted fictional play: all values are authored, none sampled."""

    play_type: str
    qtr: int
    clock: float
    down: int | None
    ydstogo: int | None
    yards_gained: float
    desc: str
    qb_spike: int = 0
    quarter_end: int = 0
    penalty: int = 0
    timeout: int = 0


# (game_id, game_date, home_team, away_team): three games before the documented
# holdout cutoff and three after it, so the split always has both sides.
_GAMES: tuple[tuple[int, str, str, str], ...] = (
    (2016091101, "2016-09-11", "NE", "BUF"),
    (2016091801, "2016-09-18", "KC", "DEN"),
    (2016100201, "2016-10-02", "SEA", "SF"),
    (2017091001, "2017-09-10", "GB", "CHI"),
    (2017100101, "2017-10-01", "DAL", "PHI"),
    (2017110501, "2017-11-05", "NO", "ATL"),
)

# One scripted game with 33 plays covering all six target labels plus filtered
# non-target rows (kickoffs, timeouts, penalties) that still provide context.
_SCRIPT: tuple[_Play, ...] = (
    _Play("kickoff", 1, 900, None, None, 60, "kickoff 60 yards"),
    _Play("pass", 1, 880, 1, 10, 8, "short pass complete"),
    _Play("run", 1, 855, 2, 2, 3, "run up the middle"),
    _Play("pass", 1, 830, 1, 10, 12, "deep pass complete"),
    _Play("pass", 1, 800, 1, 10, 0, "pass incomplete"),
    _Play("punt", 1, 780, 4, 10, 45, "punt 45 yards"),
    _Play("pass", 1, 700, 1, 10, 5, "quick pass complete"),
    _Play("no_play", 1, 660, None, None, 0, "timeout", timeout=1),
    _Play("run", 1, 640, 2, 5, -1, "run left tackle"),
    _Play("pass", 1, 610, 3, 6, 0, "pass incomplete"),
    _Play("field_goal", 1, 580, 4, 6, 0, "field goal 42 yards"),
    _Play("kickoff", 2, 900, None, None, 55, "kickoff 55 yards"),
    _Play("pass", 2, 770, 1, 10, 7, "pass complete"),
    _Play("run", 2, 740, 2, 3, 2, "run right guard"),
    _Play("run", 2, 710, 1, 10, 4, "run up the middle"),
    _Play("qb_spike", 2, 6, 1, 10, 0, "spiked the ball", qb_spike=1),
    _Play("field_goal", 2, 3, 2, 10, 0, "field goal 51 yards", quarter_end=1),
    _Play("kickoff", 3, 900, None, None, 62, "kickoff 62 yards"),
    _Play("run", 3, 870, 1, 10, 4, "run up the middle"),
    _Play("pass", 3, 840, 2, 6, 9, "pass complete"),
    _Play("run", 3, 810, 1, 10, 2, "run left end"),
    _Play("pass", 3, 780, 2, 8, 15, "deep pass complete"),
    _Play("field_goal", 3, 740, 4, 1, 0, "field goal 35 yards"),
    _Play("kickoff", 4, 900, None, None, 58, "kickoff 58 yards"),
    _Play("pass", 4, 870, 1, 10, 6, "pass complete"),
    _Play("run", 4, 845, 2, 4, -2, "run up the middle"),
    _Play("no_play", 4, 830, None, None, 0, "penalty declined", penalty=1),
    _Play("punt", 4, 810, 4, 6, 40, "punt 40 yards"),
    _Play("pass", 4, 770, 1, 10, 11, "pass complete"),
    _Play("run", 4, 740, 1, 10, 3, "run right tackle"),
    _Play("qb_spike", 4, 30, 2, 7, 0, "spiked the ball", qb_spike=1),
    _Play("qb_kneel", 4, 6, 1, 10, -1, "kneels down"),
    _Play("qb_kneel", 4, 2, 1, 10, -1, "kneels down", quarter_end=1),
)


def fictional_plays() -> pd.DataFrame:
    """Return the deterministic fictional raw play frame (no NFL rows)."""
    rows: list[dict[str, Any]] = []
    for game_index, (game_id, game_date, home_team, away_team) in enumerate(_GAMES):
        clock_shift = (game_index * 5) % 25
        for play_index, play in enumerate(_SCRIPT, start=1):
            clock = float(max(play.clock - clock_shift, 1))
            rows.append(
                {
                    "game_id": game_id,
                    "play_id": play_index,
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "posteam": home_team if play_index % 2 else away_team,
                    "defteam": away_team if play_index % 2 else home_team,
                    "yardline_100": float(40 + (play_index * 11 + game_index * 7) % 55),
                    "quarter_seconds_remaining": clock,
                    "half_seconds_remaining": clock + 900 if play.qtr in (1, 3) else clock,
                    "qtr": play.qtr,
                    "down": play.down,
                    "ydstogo": play.ydstogo,
                    "posteam_timeouts_remaining": 3 - (play_index // 12) % 4,
                    "defteam_timeouts_remaining": 3 - (play_index // 9) % 4,
                    "score_differential": float((play_index * 3 + game_index * 5) % 25 - 12),
                    "play_type": play.play_type,
                    "yards_gained": play.yards_gained,
                    "desc": play.desc,
                    "qb_spike": play.qb_spike,
                    "quarter_end": play.quarter_end,
                    "penalty": play.penalty,
                    "timeout": play.timeout,
                }
            )
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


def run_smoke(output_dir: str | Path = "artifacts/smoke") -> dict[str, Any]:
    """Run the full pipeline on the fictional fixture and write a mechanics report.

    Returns the report dict and writes ``smoke_report.json`` plus ``smoke_models.joblib``
    into ``output_dir``. Raises RuntimeError if any mechanics invariant fails.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    raw = fictional_plays()
    prepared = prepare_plays(raw)
    train, test = chronological_split(prepared, holdout_start=SMOKE_HOLDOUT_START)
    models = fit_models(train, seed=SMOKE_SEED, trees=SMOKE_TREES, jobs=SMOKE_JOBS)

    train_games = set(train["game_id"])
    test_games = set(test["game_id"])
    metadata: dict[str, Any] = {
        "purpose": "mechanics smoke test on a deterministic fictional fixture",
        "seed": SMOKE_SEED,
        "trees": SMOKE_TREES,
        "jobs": SMOKE_JOBS,
        "holdout_start": SMOKE_HOLDOUT_START,
        "raw_rows": int(len(raw)),
        "prepared_rows": int(len(prepared)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_games": len(train_games),
        "test_games": len(test_games),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_labels": list(TARGET_LABELS),
    }

    bundle_path = destination / SMOKE_BUNDLE_NAME
    save_bundle(models, bundle_path, metadata)
    bundle = load_bundle(bundle_path)
    restored = bundle["models"]

    features = test[FEATURE_COLUMNS]
    round_trip: dict[str, dict[str, bool | None]] = {}
    for name, model in models.items():
        labels_equal = bool(
            np.array_equal(model.predict(features), restored[name].predict(features))
        )
        probabilities_equal: bool | None = None
        if hasattr(model, "predict_proba") and hasattr(restored[name], "predict_proba"):
            probabilities_equal = bool(
                np.array_equal(
                    model.predict_proba(features), restored[name].predict_proba(features)
                )
            )
        round_trip[name] = {
            "labels_equal": labels_equal,
            "probabilities_equal": probabilities_equal,
        }

    train_dates = pd.to_datetime(train["game_date"])
    test_dates = pd.to_datetime(test["game_date"])
    checks: dict[str, bool] = {
        "all_documented_labels_in_train": set(train["play_type"]) == set(TARGET_LABELS),
        "all_documented_labels_in_test": set(test["play_type"]) == set(TARGET_LABELS),
        "train_dates_before_holdout": bool(train_dates.max() < pd.Timestamp(SMOKE_HOLDOUT_START)),
        "test_dates_from_holdout": bool(test_dates.min() >= pd.Timestamp(SMOKE_HOLDOUT_START)),
        "split_games_disjoint": not (train_games & test_games),
        "save_load_labels_equal": all(outcome["labels_equal"] for outcome in round_trip.values()),
        "save_load_probabilities_equal": all(
            outcome["probabilities_equal"] is not False for outcome in round_trip.values()
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"smoke mechanics checks failed: {', '.join(failed)}")

    report: dict[str, Any] = {
        "mechanics_only": True,
        "note": (
            "Mechanics-only smoke test on a deterministic fictional fixture; "
            "the metrics below verify pipeline wiring and are not performance claims."
        ),
        "fixture": {
            "description": "deterministic original fictional play-by-play; no NFL rows copied",
            "raw_rows": int(len(raw)),
            "prepared_rows": int(len(prepared)),
            "games": int(raw["game_id"].nunique()),
            "target_labels": list(TARGET_LABELS),
        },
        "split": {
            "holdout_start": SMOKE_HOLDOUT_START,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_games": len(train_games),
            "test_games": len(test_games),
            "train_min_date": str(train_dates.min().date()),
            "train_max_date": str(train_dates.max().date()),
            "test_min_date": str(test_dates.min().date()),
            "test_max_date": str(test_dates.max().date()),
        },
        "models": sorted(models),
        "round_trip": round_trip,
        "checks": checks,
        "metrics": evaluate_models(restored, test),
        "artifacts": {"bundle": SMOKE_BUNDLE_NAME, "report": SMOKE_REPORT_NAME},
    }

    report_path = destination / SMOKE_REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(report)
    return report


def _print_summary(report: dict[str, Any]) -> None:
    fixture = report["fixture"]
    split = report["split"]
    print("smoke: fictional-fixture mechanics pipeline passed")
    print(
        f"  fixture: {fixture['raw_rows']} raw rows -> {fixture['prepared_rows']} prepared rows "
        f"across {fixture['games']} games"
    )
    print(
        f"  split: {split['train_rows']} train rows / {split['test_rows']} test rows, "
        "games disjoint"
    )
    print("  save/load: labels and probabilities match for every model")
    for name in report["models"]:
        metrics = report["metrics"][name]
        print(f"  {name}: accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}")
    print("  note: metrics describe fictional-fixture mechanics only, not real-data performance")
    print(f"  wrote {report['artifacts']['report']} and {report['artifacts']['bundle']}")
