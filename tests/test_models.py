"""Consumer-visible behavior of the public models interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_showcase.features import (
    FEATURE_COLUMNS,
    TARGET_LABELS,
    chronological_split,
    prepare_plays,
)
from nfl_showcase.models import (
    evaluate_models,
    fit_models,
    load_bundle,
    save_bundle,
)
from nfl_showcase.smoke import fictional_plays

MODEL_NAMES = {"majority", "down_distance", "logistic_regression", "random_forest"}
HOLDOUT_START = "2017-09-01"


class _ConstantPassPredictor:
    """A minimal estimator that predicts one label for every row.

    Metric tests use it so the result can be checked against hand-computed
    values.
    """

    def __init__(self, label: str = "pass") -> None:
        self.label = label
        self.classes_ = np.asarray(TARGET_LABELS)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.label)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.zeros((len(features), len(self.classes_)))
        probabilities[:, list(self.classes_).index(self.label)] = 1.0
        return probabilities


def test_fit_models_predicts_documented_labels_from_raw_feature_frames(
    trained_models, fictional_split
):
    _, test = fictional_split
    features = test[FEATURE_COLUMNS]
    assert list(features.columns) == list(FEATURE_COLUMNS)

    for name, model in trained_models.items():
        predictions = np.asarray(model.predict(features))
        assert predictions.shape == (len(features),), name
        assert set(predictions) <= set(TARGET_LABELS), name


def test_probabilistic_models_return_aligned_label_probabilities(trained_models, fictional_split):
    _, test = fictional_split
    features = test[FEATURE_COLUMNS]

    for name, model in trained_models.items():
        probabilities = np.asarray(model.predict_proba(features))
        assert probabilities.shape == (len(features), len(model.classes_)), name
        assert set(model.classes_) <= set(TARGET_LABELS), name
        assert np.allclose(probabilities.sum(axis=1), 1.0), name


def test_test_only_extremes_and_unseen_categories_do_not_break_preprocessing(
    trained_models, fictional_split
):
    _, test = fictional_split
    probe = test.index[0]
    untouched = test.loc[test.index != probe, FEATURE_COLUMNS]
    outlier = test.loc[[probe], FEATURE_COLUMNS].copy()
    outlier.loc[probe, "home_team"] = "ZZZ"
    outlier.loc[probe, "away_team"] = "ZZZ"
    outlier.loc[probe, "previous_play_type"] = "never_seen_before"
    outlier.loc[probe, "yardline_100"] = 5000.0
    outlier.loc[probe, "previous_yards_gained"] = 9999.0
    outlier.loc[probe, "quarter_seconds_remaining"] = -500.0
    outlier.loc[probe, "down"] = np.nan
    combined = pd.concat([untouched, outlier])

    for name, model in trained_models.items():
        baseline_labels = np.asarray(model.predict(untouched))
        baseline_probabilities = np.asarray(model.predict_proba(untouched))
        assert np.isfinite(baseline_probabilities).all(), name

        batch_labels = np.asarray(model.predict(combined))
        assert set(batch_labels) <= set(TARGET_LABELS), name
        batch_probabilities = np.asarray(model.predict_proba(combined))
        assert np.isfinite(batch_probabilities).all(), name
        untouched_labels = batch_labels[: len(untouched)]
        assert np.array_equal(untouched_labels, baseline_labels), name
        untouched_probabilities = batch_probabilities[: len(untouched)]
        assert np.allclose(
            untouched_probabilities, baseline_probabilities, rtol=1e-9, atol=1e-12
        ), name

        single_probabilities = np.asarray(model.predict_proba(outlier))
        probe_probabilities = batch_probabilities[len(untouched)]
        assert np.allclose(single_probabilities[0], probe_probabilities, rtol=1e-9, atol=1e-12), (
            name
        )

        # Inference must not mutate the fitted model.
        repeated_labels = np.asarray(model.predict(untouched))
        assert np.array_equal(repeated_labels, baseline_labels), name
        repeated_probabilities = np.asarray(model.predict_proba(untouched))
        assert np.allclose(repeated_probabilities, baseline_probabilities, rtol=1e-9, atol=1e-12), (
            name
        )


def test_evaluate_models_uses_ground_truth_and_documented_label_order(fictional_split):
    _, test = fictional_split
    mixed = pd.concat(
        [
            test[test["play_type"] == "pass"].head(3),
            test[test["play_type"] == "run"].head(3),
        ]
    )
    assert len(mixed) == 6

    metrics = evaluate_models({"constant_pass": _ConstantPassPredictor()}, mixed)

    report = metrics["constant_pass"]
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["macro_f1"] == pytest.approx(1.0 / 9.0)
    assert report["per_class"]["pass"]["support"] == 3
    assert report["per_class"]["run"]["support"] == 3
    assert report["per_class"]["qb_spike"]["support"] == 0
    assert report["per_class"]["qb_spike"]["f1-score"] == 0.0
    matrix = np.asarray(report["confusion_matrix"])
    pass_index = list(TARGET_LABELS).index("pass")
    run_index = list(TARGET_LABELS).index("run")
    assert matrix[pass_index].tolist() == [0, 3, 0, 0, 0, 0]
    assert matrix[run_index].tolist() == [0, 3, 0, 0, 0, 0]


def test_evaluate_models_preserves_labels_absent_from_the_test_frame(
    trained_models, fictional_split
):
    _, test = fictional_split
    without_spikes = test[test["play_type"] != "qb_spike"]
    assert len(without_spikes) > 0

    metrics = evaluate_models(trained_models, without_spikes)

    for name, report in metrics.items():
        assert report["per_class"]["qb_spike"]["support"] == 0, name
        assert list(report["labels"]) == list(TARGET_LABELS), name


def test_save_and_load_round_trip_preserves_labels_and_probabilities(
    trained_models, fictional_split, tmp_path
):
    _, test = fictional_split
    features = test[FEATURE_COLUMNS]
    bundle_path = tmp_path / "nested" / "models.joblib"
    metadata = {"seed": 42, "trees": 8, "note": "round-trip test"}

    save_bundle(trained_models, bundle_path, metadata)

    assert bundle_path.exists()
    bundle = load_bundle(bundle_path)
    assert set(bundle) == {"models", "metadata"}
    assert bundle["metadata"] == metadata
    assert set(bundle["models"]) == MODEL_NAMES
    for name, original in trained_models.items():
        restored = bundle["models"][name]
        expected_labels = original.predict(features)
        actual_labels = restored.predict(features)
        assert np.array_equal(expected_labels, actual_labels), name
        expected_probabilities = original.predict_proba(features)
        actual_probabilities = restored.predict_proba(features)
        assert np.array_equal(expected_probabilities, actual_probabilities), name


def test_fit_models_is_deterministic_for_a_fixed_seed(fictional_split):
    train, test = fictional_split
    features = test[FEATURE_COLUMNS]

    first = fit_models(train, seed=42, trees=8, jobs=1)
    second = fit_models(train, seed=42, trees=8, jobs=1)

    assert set(first) == MODEL_NAMES
    for name in first:
        expected_labels = first[name].predict(features)
        actual_labels = second[name].predict(features)
        assert np.array_equal(expected_labels, actual_labels), name
        expected_probabilities = first[name].predict_proba(features)
        actual_probabilities = second[name].predict_proba(features)
        assert np.array_equal(expected_probabilities, actual_probabilities), name


def _interleaved(raw: pd.DataFrame, future: pd.Series) -> pd.DataFrame:
    """Return ``raw`` rows reversed and alternating between past and future."""
    past = raw.loc[~future].iloc[::-1]
    future_rows = raw.loc[future].iloc[::-1]
    frames: list[pd.DataFrame] = []
    for position in range(max(len(past), len(future_rows))):
        if position < len(future_rows):
            frames.append(future_rows.iloc[[position]])
        if position < len(past):
            frames.append(past.iloc[[position]])
    return pd.concat(frames, ignore_index=True)


def test_future_games_cannot_change_transforms_learned_from_the_past():
    raw = fictional_plays()
    future = pd.to_datetime(raw["game_date"]) >= pd.Timestamp(HOLDOUT_START)
    assert future.any() and not future.all()

    altered = raw.copy()
    altered.loc[future, ["home_team", "away_team", "posteam", "defteam"]] = "ZZZ"
    altered.loc[future, "yardline_100"] = 9999.0
    altered.loc[future, "quarter_seconds_remaining"] = -5000.0
    altered.loc[future, "half_seconds_remaining"] = -5000.0
    altered.loc[future, "yards_gained"] = 9999
    altered.loc[future, "score_differential"] = 9999.0

    baseline_train, _ = chronological_split(
        prepare_plays(_interleaved(raw, future)), holdout_start=HOLDOUT_START
    )
    altered_train, altered_test = chronological_split(
        prepare_plays(_interleaved(altered, future)), holdout_start=HOLDOUT_START
    )

    pd.testing.assert_frame_equal(baseline_train, altered_train)

    baseline_models = fit_models(baseline_train, seed=42, trees=8, jobs=1)
    altered_models = fit_models(altered_train, seed=42, trees=8, jobs=1)

    probe = altered_test[FEATURE_COLUMNS]
    assert len(probe) > 0
    for name, baseline_model in baseline_models.items():
        altered_model = altered_models[name]
        expected_labels = baseline_model.predict(probe)
        actual_labels = altered_model.predict(probe)
        assert np.array_equal(expected_labels, actual_labels), name
        expected_probabilities = baseline_model.predict_proba(probe)
        actual_probabilities = altered_model.predict_proba(probe)
        assert np.array_equal(expected_probabilities, actual_probabilities), name
