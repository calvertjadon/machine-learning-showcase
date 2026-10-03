"""Original machine learning study that predicts NFL play calls.

This package reimplements from scratch the feature engineering and modelling
workflow of a 2024 course project about pre-snap play-call prediction.  The
code is original, and it fixes the methodological problems of the historical
experiment rather than reproducing them.  Those problems were leakage from
random splits, features shifted across game boundaries, and post-play
information used as input.

Modules:

* :mod:`nfl_showcase.features` builds causal features and splits the data
  chronologically into training and holdout partitions, keeping every game
  whole.
* :mod:`nfl_showcase.models` builds, fits, evaluates and persists the models.
* :mod:`nfl_showcase.cli` provides the command line entry points ``train``,
  ``evaluate``, ``predict`` and ``smoke``.

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
