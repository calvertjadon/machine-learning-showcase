"""Original machine-learning showcase: predicting NFL play calls.

This package reimplements, from scratch, the feature engineering and modelling
workflow of a 2024 course project about pre-snap play-call prediction.  It is
original code: it fixes the methodological issues of the historical
experiment (leakage from random splits, features shifted across game
boundaries, post-play information used as input) instead of reproducing them.

Modules:

* :mod:`nfl_showcase.features` -- causal feature engineering plus a
  game-separated chronological train/holdout split.
* :mod:`nfl_showcase.models` -- model construction, fitting, evaluation and
  persistence.
* :mod:`nfl_showcase.cli` -- command line entry points (``train``,
  ``evaluate``, ``predict``, ``smoke``).

The features public API is re-exported here for convenience::

    from nfl_showcase import FEATURE_COLUMNS, chronological_split, clean_raw_plays, prepare_plays
"""

from nfl_showcase.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    TARGET_LABELS,
    chronological_split,
    clean_raw_plays,
    prepare_plays,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CATEGORICAL_FEATURES",
    "FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "RAW_COLUMNS",
    "TARGET_LABELS",
    "chronological_split",
    "clean_raw_plays",
    "prepare_plays",
]
