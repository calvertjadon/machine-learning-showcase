"""Fitted estimators, evaluation and bundle persistence for the showcase.

Public interface
----------------
``build_models(seed=42, trees=200, jobs=2)``
    Unfitted estimators keyed ``"majority"``, ``"down_distance"``,
    ``"logistic_regression"`` and ``"random_forest"``.
``fit_models(train, seed=42, trees=200, jobs=2)``
    The same dictionary with every estimator fitted on ``train``.
``evaluate_models(models, test)``
    JSON-serializable accuracy, macro F1, per-class report, confusion matrix
    and explicit label order for each model.
``save_bundle(models, path, metadata)`` / ``load_bundle(path)``
    Joblib persistence of complete fitted pipelines plus provenance metadata.

Every learned transform (median imputation and standard scaling for numeric
features, most-frequent imputation and one-hot encoding for categorical
features) lives inside the estimators, is fitted only on the training frame
and therefore travels with a saved bundle.  Categorical columns may arrive as
object columns with ``numpy.nan`` gaps (what ``prepare_plays`` produces) or as
pandas extension strings with ``pd.NA``; the categorical pipeline normalizes
the latter to ``numpy.nan`` before imputation, which scikit-learn 1.5
imputers require.  Saved bundles are trusted local artifacts: loading one
unpickles Python objects, so only load bundles this package produced.  A
bundle written by a different scikit-learn version may fail to load or behave
differently because pickles embed version-sensitive internals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_LABELS,
)

#: Label column carried by prepared frames.
TARGET_COLUMN = "play_type"

#: Estimator keys returned by :func:`build_models`, in report order.
MODEL_NAMES = ("majority", "down_distance", "logistic_regression", "random_forest")

FOREST_MAX_DEPTH = 18
FOREST_MIN_SAMPLES_LEAF = 5
LOGISTIC_MAX_ITER = 1000


def _require_frame(frame: pd.DataFrame, required: Sequence[str], what: str) -> pd.DataFrame:
    """Reject anything that is not a non-empty frame carrying ``required``."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a pandas DataFrame, got {type(frame).__name__}")
    if len(frame) == 0:
        raise ValueError(f"{what} contains no rows")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing required columns: {', '.join(missing)}")
    return frame


def _modal_label(counts: Counter) -> str:
    """Most frequent label; ties break toward the alphabetically first label."""
    if not counts:
        raise ValueError("cannot pick a modal label from an empty collection")
    best = max(counts.values())
    return str(min(label for label, count in counts.items() if count == best))


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars and containers to plain JSON-serializable types."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list) or isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _macro_f1_score(truth, predictions, labels: Sequence[str]) -> float:
    """Macro F1 over explicit labels, scoring unseen classes as zero."""
    return float(f1_score(truth, predictions, labels=labels, average="macro", zero_division=0))


class DownDistanceBaseline(ClassifierMixin, BaseEstimator):
    """Modal play type per (down, yards-to-go bucket) with a prior fallback.

    Buckets follow the usual coaching shorthand: ``<=2``, ``<=5``, ``<=10`` and
    ``>10`` yards to go.  Training learns the play-type distribution of every
    observed ``(down, bucket)`` cell; :meth:`predict` returns that cell's modal
    play type and :meth:`predict_proba` returns its empirical training
    distribution.  Rows whose down or distance is missing, and cell/down
    combinations never seen in training, fall back to the training set's
    overall label distribution.  The estimator consumes the raw
    ``FEATURE_COLUMNS`` frame and reads only ``down`` and ``ydstogo``; no
    outcome-based rules are involved.
    """

    DISTANCE_BUCKETS = ((2.0, "<=2"), (5.0, "<=5"), (10.0, "<=10"))

    def __init__(self, down_column: str = "down", distance_column: str = "ydstogo"):
        self.down_column = down_column
        self.distance_column = distance_column

    def fit(self, X, y):
        """Learn per-cell label counts and the overall training prior."""
        frame = _require_frame(
            X,
            [self.down_column, self.distance_column],
            "down-distance training features",
        )
        target = pd.Series(y, dtype="object").reset_index(drop=True)
        if len(target) != len(frame):
            raise ValueError("X and y have inconsistent lengths")
        if target.isna().any():
            raise ValueError("down-distance targets contain missing labels")
        self.classes_ = np.sort(np.unique(target.to_numpy(dtype=object)))
        self.prior_ = Counter(target.tolist())
        self.majority_ = _modal_label(self.prior_)
        table: dict[tuple[int, str], Counter] = {}
        for down, distance, label in zip(
            frame[self.down_column], frame[self.distance_column], target, strict=True
        ):
            key = self._bucket_key(down, distance)
            if key is None:
                continue
            table.setdefault(key, Counter())[label] += 1
        self.table_ = table
        return self

    def predict(self, X):
        """Predict the modal play type of each row's down-and-distance cell."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        """Empirical training label distribution of each row's cell.

        Cells unseen in training (or rows missing down/distance) use the
        training set's overall label distribution, so the rows still sum to 1
        instead of inventing a value.
        """
        check_is_fitted(self, ["classes_", "table_", "prior_"])
        frame = _require_frame(
            X,
            [self.down_column, self.distance_column],
            "down-distance prediction features",
        )
        probabilities = np.empty((len(frame), len(self.classes_)), dtype=float)
        for row, (down, distance) in enumerate(
            zip(frame[self.down_column], frame[self.distance_column], strict=True)
        ):
            key = self._bucket_key(down, distance)
            counts = self.prior_ if key is None else self.table_.get(key, self.prior_)
            probabilities[row] = self._distribution(counts)
        return probabilities

    def _distribution(self, counts: Counter) -> np.ndarray:
        """Normalize label counts into ``classes_`` order."""
        total = float(sum(counts.values()))
        return np.array(
            [counts.get(label, 0) / total for label in self.classes_],
            dtype=float,
        )

    def _bucket_key(self, down: Any, distance: Any) -> tuple[int, str] | None:
        """Map raw down/distance into a hashable cell key, or ``None``."""
        if pd.isna(down) or pd.isna(distance):
            return None
        try:
            down_value = int(down)
            distance_value = float(distance)
        except (TypeError, ValueError):
            return None
        for threshold, bucket in self.DISTANCE_BUCKETS:
            if distance_value <= threshold:
                return (down_value, bucket)
        return (down_value, ">10")


def _numeric_transformer() -> Pipeline:
    """Median imputation then standard scaling, fitted on training rows only."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _as_object_frame(values) -> pd.DataFrame:
    """Object-dtype frame with ``numpy.nan`` marking missing values.

    Categorical columns produced by ``prepare_plays`` already satisfy this
    (object dtype, ``numpy.nan`` gaps); pandas extension string columns use
    ``pd.NA``, which scikit-learn 1.5 imputers cannot mask.  Normalizing to
    ``numpy.nan`` keeps the same missing semantics in a representation the
    imputers support, so the pipelines accept either producer dtype.
    """
    frame = pd.DataFrame(values)
    return frame.astype(object).where(frame.notna(), np.nan)


def _categorical_transformer() -> Pipeline:
    """Most-frequent imputation then one-hot encoding that ignores unknowns."""
    return Pipeline(
        [
            (
                "as_object",
                FunctionTransformer(
                    _as_object_frame,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def _preprocessor() -> ColumnTransformer:
    """Raw ``FEATURE_COLUMNS`` frame to numeric design matrix, train-fitted."""
    return ColumnTransformer(
        [
            ("numeric", _numeric_transformer(), list(NUMERIC_FEATURES)),
            ("categorical", _categorical_transformer(), list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(seed: int = 42, trees: int = 200, jobs: int = 2) -> dict[str, Any]:
    """Return the four unfitted estimators used by the showcase.

    ``majority`` predicts the most frequent play type and ``down_distance``
    predicts the modal play type of its down and yards-to-go bucket.  The two
    pipelines fit median/std numeric and most-frequent/one-hot categorical
    transforms on the training frame only.  The forest is bounded
    (``max_depth=18``, ``min_samples_leaf=5``) and seeded; no hyperparameter
    search runs here.
    """
    if trees < 1:
        raise ValueError("trees must be at least 1")
    if jobs == 0:
        raise ValueError("jobs must not be 0")
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "down_distance": DownDistanceBaseline(),
        "logistic_regression": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=seed),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=trees,
                        max_depth=FOREST_MAX_DEPTH,
                        min_samples_leaf=FOREST_MIN_SAMPLES_LEAF,
                        random_state=seed,
                        n_jobs=jobs,
                    ),
                ),
            ]
        ),
    }


def fit_models(
    train: pd.DataFrame, seed: int = 42, trees: int = 200, jobs: int = 2
) -> dict[str, Any]:
    """Fit the four showcase estimators on ``train`` and return them.

    ``train`` is a prepared frame carrying ``FEATURE_COLUMNS`` and the
    ``play_type`` label column; learned transforms see only these rows.  Labels
    must be non-missing strings drawn from ``TARGET_LABELS``.
    """
    frame = _require_frame(train, [*FEATURE_COLUMNS, TARGET_COLUMN], "training frame")
    labels = frame[TARGET_COLUMN]
    if labels.isna().any():
        raise ValueError("training frame contains missing play_type labels")
    unknown = sorted(set(labels.astype(str).unique()) - set(TARGET_LABELS))
    if unknown:
        raise ValueError(f"training frame contains unknown play_type labels: {unknown}")
    if labels.nunique() < 2:
        raise ValueError("training frame must contain at least two play_type labels")

    models = build_models(seed=seed, trees=trees, jobs=jobs)
    features = frame[list(FEATURE_COLUMNS)]
    target = labels.astype(str)
    for model in models.values():
        model.fit(features, target)
    return models


def evaluate_models(models: Mapping[str, Any], test: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Score every model against ``test`` with explicit ``TARGET_LABELS``.

    Ground truth comes first and labels are fixed to ``TARGET_LABELS`` (ground
    truth order), so rare classes keep their rows in the report even when a
    model never predicts them.  ``zero_division=0`` keeps every metric finite.
    """
    frame = _require_frame(test, [*FEATURE_COLUMNS, TARGET_COLUMN], "test frame")
    if not models:
        raise ValueError("no models were provided for evaluation")
    truth = frame[TARGET_COLUMN]
    if truth.isna().any():
        raise ValueError("test frame contains missing play_type labels")
    unknown = sorted(set(truth.astype(str).unique()) - set(TARGET_LABELS))
    if unknown:
        raise ValueError(f"test frame contains unknown play_type labels: {unknown}")
    truth = truth.astype(str)
    features = frame[list(FEATURE_COLUMNS)]
    labels = list(TARGET_LABELS)
    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        predictions = np.asarray(model.predict(features)).astype(str)
        report = classification_report(
            truth, predictions, labels=labels, zero_division=0, output_dict=True
        )
        matrix = confusion_matrix(truth, predictions, labels=labels).astype(int)
        results[str(name)] = {
            "accuracy": float(accuracy_score(truth, predictions)),
            "macro_f1": _macro_f1_score(truth, predictions, labels),
            "per_class": _jsonable(report),
            "confusion_matrix": matrix.tolist(),
            "labels": labels,
        }
    return results


def save_bundle(
    models: Mapping[str, Any],
    path: str | Path,
    metadata: Mapping[str, Any],
) -> None:
    """Write fitted models and provenance metadata to ``path`` with joblib.

    Missing parent directories are created.  The bundle is a pickle meant for
    local, trusted use only: loading it executes Python objects.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": dict(models), "metadata": dict(metadata)}, target)


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load a bundle written by :func:`save_bundle`.

    Returns the persisted ``{"models": ..., "metadata": ...}`` payload.  Only
    local, trusted bundles should be loaded; bundles written by a different
    scikit-learn version may load but score differently because pickles embed
    version-sensitive internals.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model bundle not found: {source}")
    try:
        payload = joblib.load(source)
    except Exception as error:
        raise ValueError(f"{source} could not be loaded as a joblib bundle: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{source} is not a model bundle written by save_bundle")
    if "models" not in payload or "metadata" not in payload:
        raise ValueError(
            f"{source} is not a model bundle written by save_bundle"
            " (expected 'models' and 'metadata' keys)"
        )
    models = payload["models"]
    metadata = payload["metadata"]
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{source} bundle contains no fitted models")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source} bundle metadata is not a mapping")
    return {"models": models, "metadata": metadata}
