"""Command line interface for the NFL play-type showcase.

Commands
--------
``train``
    Clean exact duplicate rows (optionally excluding whole ambiguous games),
    fit the four estimators on a chronological split, and save the bundle plus
    a provenance record.
``evaluate``
    Score a saved bundle on its own recorded holdout and write aggregate
    reports (JSON metrics plus original figures).
``predict``
    Score explicit pre-play contexts with the saved random forest and print
    JSON labels with probabilities.
``smoke``
    Run the deterministic fictional-data mechanics check from ``smoke.py``.

Every command is a plain function taking an ``argparse.Namespace`` and
returning an exit status, so the module stays importable and testable without
invoking a process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from .features import (
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    TARGET_LABELS,
    chronological_split,
    clean_raw_plays,
    prepare_plays,
)
from .models import (
    FOREST_MAX_DEPTH,
    FOREST_MIN_SAMPLES_LEAF,
    MODEL_NAMES,
    evaluate_models,
    fit_models,
    load_bundle,
    save_bundle,
)

DEFAULT_HOLDOUT_START = "2017-09-01"
DEFAULT_TRAIN_DIR = "artifacts"
DEFAULT_MODEL_PATH = "artifacts/models.joblib"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_SMOKE_DIR = "artifacts/smoke"


def _read_raw_csv(path: Path) -> pd.DataFrame:
    """Read only the minimal raw context columns (the CSV carries a UTF-8 BOM)."""
    if not path.is_file():
        raise FileNotFoundError(f"raw play-by-play csv not found: {path}")
    return pd.read_csv(
        path,
        usecols=list(RAW_COLUMNS),
        encoding="utf-8-sig",
        low_memory=False,
    )


def _sha256_of_file(path: Path) -> str:
    """Full sha256 of a file, streamed so the raw CSV is never loaded twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_summary(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    """Flat provenance summary for one side of the chronological split."""
    dates = pd.to_datetime(frame["game_date"])
    counts = frame["play_type"].value_counts()
    return {
        f"{prefix}_rows": int(len(frame)),
        f"{prefix}_games": int(frame["game_id"].nunique()),
        f"{prefix}_date_min": dates.min().date().isoformat(),
        f"{prefix}_date_max": dates.max().date().isoformat(),
        f"{prefix}_label_counts": {label: int(counts.get(label, 0)) for label in TARGET_LABELS},
    }


def _print_cleaning_audit(summary: dict[str, Any]) -> None:
    """Short audit line for the exact-duplicate and ambiguous-game decisions."""
    print(
        "cleaning: "
        f"raw_rows={summary['raw_rows']} "
        f"exact_duplicate_rows_removed={summary['exact_duplicate_rows_removed']} "
        f"ambiguous_id_groups={summary['ambiguous_id_groups']} "
        f"ambiguous_games_excluded={summary['ambiguous_games_excluded']} "
        f"ambiguous_rows_removed={summary['ambiguous_rows_removed']} "
        f"retained_rows={summary['retained_rows']}"
    )


def _nan_for_null(value: Any) -> Any:
    """Map JSON ``null`` to NaN so the saved imputers treat it as missing."""
    return np.nan if value is None else value


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a raw ``FEATURE_COLUMNS`` frame, one row per input record.

    Nulls fall through to the bundle's train-fitted imputers; this module never
    substitutes a value of its own for missing pre-play context.
    """
    rows = []
    for record in records:
        rows.append({column: _nan_for_null(record[column]) for column in FEATURE_COLUMNS})
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))


def _validate_models(models: dict[str, Any], frame: pd.DataFrame) -> None:
    """Exercise every fitted model on the holdout before its bundle is written."""
    features = frame[list(FEATURE_COLUMNS)]
    for name in MODEL_NAMES:
        model = models[name]
        predictions = np.asarray(model.predict(features)).astype(str)
        if len(predictions) != len(frame):
            raise ValueError(
                f"model {name} returned {len(predictions)} predictions for {len(frame)} rows"
            )
        unexpected = sorted(set(predictions) - set(TARGET_LABELS))
        if unexpected:
            raise ValueError(f"model {name} predicted unexpected labels: {unexpected}")
        if hasattr(model, "predict_proba"):
            sample = features.head(200)
            probabilities = np.asarray(model.predict_proba(sample))
            class_count = len(np.asarray(model.classes_))
            if probabilities.shape != (len(sample), class_count):
                raise ValueError(
                    f"model {name} returned probabilities with shape {probabilities.shape}"
                )
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"model {name} probability rows do not sum to 1")


def _check_reconstructed_split(metadata: dict[str, Any], prefix: str, frame: pd.DataFrame) -> None:
    """Refuse to score a reconstruction that differs from the saved bundle."""
    expected = metadata.get(f"{prefix}_rows")
    if expected is not None and int(expected) != len(frame):
        raise ValueError(
            f"reconstructed {prefix} split has {len(frame)} rows but the bundle "
            f"metadata records {expected}"
        )


def _print_metrics_table(metrics: dict[str, Any]) -> None:
    """Compact measured table so a run is readable without opening the JSON."""
    width = max(len(name) for name in metrics)
    header = f"{'model':<{width}}  {'accuracy':>8}  {'macro_f1':>8}"
    print(header)
    print("-" * len(header))
    for name, values in metrics.items():
        print(f"{name:<{width}}  {values['accuracy']:>8.4f}  {values['macro_f1']:>8.4f}")


def _pyplot() -> Any:
    """Import matplotlib lazily with the headless Agg backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _plot_comparison(metrics: dict[str, Any], path: Path) -> None:
    """Grouped accuracy and macro-F1 bars for every evaluated model."""
    plt = _pyplot()
    names = list(metrics)
    accuracy = [metrics[name]["accuracy"] for name in names]
    macro_f1 = [metrics[name]["macro_f1"] for name in names]
    positions = np.arange(len(names))
    width = 0.38
    figure, axes = plt.subplots(figsize=(8.0, 4.6))
    bars_a = axes.bar(positions - width / 2, accuracy, width, label="accuracy", color="#4c72b0")
    bars_f = axes.bar(positions + width / 2, macro_f1, width, label="macro F1", color="#dd8452")
    axes.set_xticks(positions, names, rotation=15, ha="right")
    axes.set_ylim(0.0, 1.0)
    axes.set_ylabel("holdout score")
    axes.set_title("Holdout accuracy and macro F1 by model")
    axes.grid(axis="y", alpha=0.3)
    axes.legend()
    for bars in (bars_a, bars_f):
        axes.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_confusion(metrics: dict[str, Any], path: Path) -> None:
    """Holdout confusion matrix of the random forest in TARGET_LABELS order."""
    plt = _pyplot()
    forest = metrics["random_forest"]
    labels = list(forest["labels"])
    matrix = np.asarray(forest["confusion_matrix"], dtype=float)
    figure, axes = plt.subplots(figsize=(6.8, 5.6))
    image = axes.imshow(matrix, cmap="viridis")
    axes.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axes.set_yticks(range(len(labels)), labels)
    axes.set_xlabel("predicted play type")
    axes.set_ylabel("true play type")
    axes.set_title("Random forest holdout confusion matrix")
    cutoff = matrix.max() * 0.6
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            count = matrix[row, column]
            axes.text(
                column,
                row,
                f"{count:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if count > cutoff else "black",
            )
    figure.colorbar(image, ax=axes, shrink=0.85, label="plays")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_per_class_f1(metrics: dict[str, Any], path: Path) -> None:
    """Per-class holdout F1 bars, grouped by model, in TARGET_LABELS order."""
    plt = _pyplot()
    names = list(metrics)
    labels = list(next(iter(metrics.values()))["labels"])
    positions = np.arange(len(labels))
    width = 0.8 / len(names)
    offset = (len(names) - 1) / 2
    figure, axes = plt.subplots(figsize=(9.2, 4.8))
    for index, name in enumerate(names):
        values = [metrics[name]["per_class"][label]["f1-score"] for label in labels]
        axes.bar(positions + (index - offset) * width, values, width, label=name)
    axes.set_xticks(positions, labels, rotation=15, ha="right")
    axes.set_ylim(0.0, 1.0)
    axes.set_ylabel("holdout F1")
    axes.set_title("Per-class holdout F1 by model")
    axes.grid(axis="y", alpha=0.3)
    axes.legend(ncols=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_forest_importance(models: dict[str, Any], path: Path) -> None:
    """Top forest importances with an explicit impurity-importance caveat."""
    plt = _pyplot()
    forest = models["random_forest"]
    steps = getattr(forest, "named_steps", {})
    if "preprocess" not in steps or "classifier" not in steps:
        raise ValueError("random_forest is not the expected preprocessing pipeline")
    feature_names = np.asarray(steps["preprocess"].get_feature_names_out())
    importances = np.asarray(steps["classifier"].feature_importances_)
    if feature_names.shape[0] != importances.shape[0]:
        raise ValueError("forest importances do not align with transformed features")
    top = min(15, importances.shape[0])
    order = np.argsort(importances)[::-1][:top][::-1]
    figure, axes = plt.subplots(figsize=(8.4, 6.0))
    axes.barh(feature_names[order], importances[order], color="#55a868")
    axes.set_xlabel("impurity importance (mean decrease in impurity)")
    axes.set_title(
        "Random forest impurity importances\n"
        "(biased toward high-cardinality features; descriptive only)"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_train(args: argparse.Namespace) -> int:
    """Clean the raw rows, fit the models, validate them, then persist the run."""
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    raw = _read_raw_csv(data_path)
    clean_raw, cleaning_summary = clean_raw_plays(
        raw, exclude_ambiguous_games=args.exclude_ambiguous_games
    )
    prepared = prepare_plays(clean_raw)
    train_frame, test_frame = chronological_split(prepared, holdout_start=args.holdout_start)
    models = fit_models(train_frame, seed=args.seed, trees=args.trees, jobs=args.jobs)
    _validate_models(models, test_frame)
    metadata: dict[str, Any] = {
        "source_file": data_path.name,
        "source_bytes": int(data_path.stat().st_size),
        "source_sha256": _sha256_of_file(data_path),
        "exclude_ambiguous_games": bool(args.exclude_ambiguous_games),
        "source_cleaning": cleaning_summary,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "holdout_start": str(args.holdout_start),
        "seed": int(args.seed),
        "trees": int(args.trees),
        "jobs": int(args.jobs),
        "forest_params": {
            "n_estimators": int(args.trees),
            "max_depth": FOREST_MAX_DEPTH,
            "min_samples_leaf": FOREST_MIN_SAMPLES_LEAF,
            "random_state": int(args.seed),
            "n_jobs": int(args.jobs),
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "target_labels": list(TARGET_LABELS),
        **_split_summary(train_frame, "train"),
        **_split_summary(test_frame, "test"),
    }
    bundle_path = output_dir / "models.joblib"
    training_path = output_dir / "training.json"
    save_bundle(models, bundle_path, metadata)
    training_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"read {metadata['source_file']} ({metadata['source_bytes']} bytes)")
    print(f"sha256 {metadata['source_sha256']}")
    _print_cleaning_audit(cleaning_summary)
    counts = ", ".join(
        f"{label}={count}" for label, count in metadata["train_label_counts"].items()
    )
    print(f"train label counts: {counts}")
    for prefix in ("train", "test"):
        rows = metadata[f"{prefix}_rows"]
        games = metadata[f"{prefix}_games"]
        start = metadata[f"{prefix}_date_min"]
        end = metadata[f"{prefix}_date_max"]
        print(f"{prefix}: rows={rows} games={games} dates={start}..{end}")
    print(f"validated {len(models)} models on {len(test_frame)} holdout rows")
    print(f"wrote {bundle_path}")
    print(f"wrote {training_path}")
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    """Score the saved bundle on its own holdout, replaying its saved cleaning policy."""
    data_path = Path(args.data)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    bundle = load_bundle(model_path)
    models = bundle["models"]
    metadata = bundle["metadata"]
    missing_models = [name for name in MODEL_NAMES if name not in models]
    if missing_models:
        raise ValueError(f"bundle is missing fitted models: {', '.join(missing_models)}")
    recorded_sha256 = metadata.get("source_sha256")
    if not recorded_sha256:
        raise ValueError("bundle metadata does not record source_sha256")
    actual_sha256 = _sha256_of_file(data_path)
    if actual_sha256 != recorded_sha256:
        raise ValueError(
            "dataset sha256 does not match the bundle metadata "
            f"({actual_sha256} != {recorded_sha256})"
        )
    holdout_start = metadata.get("holdout_start")
    if not holdout_start:
        raise ValueError("bundle metadata does not record holdout_start")
    saved_exclude = metadata.get("exclude_ambiguous_games")
    if not isinstance(saved_exclude, bool):
        raise ValueError("bundle metadata does not record exclude_ambiguous_games as a boolean")
    recorded_cleaning = metadata.get("source_cleaning")
    if not isinstance(recorded_cleaning, dict):
        raise ValueError("bundle metadata does not record source_cleaning")
    raw = _read_raw_csv(data_path)
    clean_raw, cleaning_summary = clean_raw_plays(raw, exclude_ambiguous_games=saved_exclude)
    if cleaning_summary != recorded_cleaning:
        raise ValueError(
            "reconstructed source cleaning does not match the bundle metadata "
            f"({cleaning_summary} != {recorded_cleaning})"
        )
    _print_cleaning_audit(cleaning_summary)
    prepared = prepare_plays(clean_raw)
    train_frame, test_frame = chronological_split(prepared, holdout_start=holdout_start)
    _check_reconstructed_split(metadata, "train", train_frame)
    _check_reconstructed_split(metadata, "test", test_frame)
    metrics = evaluate_models(models, test_frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"metadata": metadata, "models": metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    comparison_path = output_dir / "comparison.png"
    confusion_path = output_dir / "random_forest_confusion.png"
    per_class_path = output_dir / "per_class_f1.png"
    importance_path = output_dir / "random_forest_importance.png"
    _plot_comparison(metrics, comparison_path)
    _plot_confusion(metrics, confusion_path)
    _plot_per_class_f1(metrics, per_class_path)
    _plot_forest_importance(models, importance_path)
    _print_metrics_table(metrics)
    print(f"wrote {metrics_path}")
    print(f"wrote {comparison_path}")
    print(f"wrote {confusion_path}")
    print(f"wrote {per_class_path}")
    print(f"wrote {importance_path}")
    return 0


def run_predict(args: argparse.Namespace) -> int:
    """Score explicit pre-play contexts with the saved random forest."""
    model_path = Path(args.model)
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"input json not found: {input_path}")
    bundle = load_bundle(model_path)
    forest = bundle["models"].get("random_forest")
    if forest is None:
        raise ValueError(f"{model_path} does not contain a fitted random_forest model")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    if not records:
        raise ValueError("input json contains no play contexts")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"input record {index} is not a JSON object")
    missing: set[str] = set()
    for record in records:
        missing.update(column for column in FEATURE_COLUMNS if column not in record)
    if missing:
        raise ValueError(
            "input records are missing required feature context: " + ", ".join(sorted(missing))
        )
    frame = _records_to_frame(records)
    predictions = forest.predict(frame)
    probabilities = forest.predict_proba(frame)
    class_names = [str(label) for label in forest.classes_]
    results = []
    for label, row in zip(predictions, probabilities, strict=True):
        row_probs = {name: float(value) for name, value in zip(class_names, row, strict=True)}
        results.append({"label": str(label), "probabilities": row_probs})
    output = results if isinstance(payload, list) else results[0]
    print(json.dumps(output, indent=2))
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    """Delegate the deterministic mechanics check to ``nfl_showcase.smoke``."""
    from .smoke import run_smoke as run_smoke_mechanics

    output_dir = Path(args.output_dir)
    report = run_smoke_mechanics(output_dir)
    checks = report.get("checks", {}) if isinstance(report, dict) else {}
    if checks:
        passed = sum(1 for value in checks.values() if value)
        print(f"smoke mechanics checks passed: {passed}/{len(checks)}")
    print(f"smoke artifacts written under {output_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nfl-showcase",
        description=(
            "Train, evaluate and apply explainable play-type models on a raw NFL play-by-play CSV."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="fit models on a chronological split and save a bundle",
    )
    train.add_argument(
        "--data",
        required=True,
        help="raw play-by-play CSV; only the minimal context columns are read",
    )
    train.add_argument(
        "--output-dir",
        default=DEFAULT_TRAIN_DIR,
        help="artifacts directory for models.joblib and training.json",
    )
    train.add_argument(
        "--holdout-start",
        default=DEFAULT_HOLDOUT_START,
        help="first date of the chronological holdout (YYYY-MM-DD)",
    )
    train.add_argument("--seed", type=int, default=42, help="random seed")
    train.add_argument("--trees", type=int, default=200, help="forest tree count")
    train.add_argument("--jobs", type=int, default=2, help="parallel forest jobs")
    train.add_argument(
        "--exclude-ambiguous-games",
        action="store_true",
        default=False,
        help=(
            "opt in to dropping every game whose (game_id, play_id) rows still carry "
            "conflicting pre-play context after exact-duplicate removal; the default "
            "rejects such a source with an actionable error"
        ),
    )
    train.set_defaults(handler=run_train)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="score the saved bundle on its own holdout",
    )
    evaluate.add_argument(
        "--data",
        required=True,
        help="the same raw CSV that trained the bundle",
    )
    evaluate.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="saved models.joblib bundle",
    )
    evaluate.add_argument(
        "--output-dir",
        default=DEFAULT_RESULTS_DIR,
        help="directory for metrics.json and the figures",
    )
    evaluate.set_defaults(handler=run_evaluate)

    predict = subparsers.add_parser(
        "predict",
        help="predict play types for explicit pre-play contexts",
    )
    predict.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="saved models.joblib bundle",
    )
    predict.add_argument(
        "--input",
        required=True,
        help="JSON object or list of objects with all feature columns",
    )
    predict.set_defaults(handler=run_predict)

    smoke = subparsers.add_parser(
        "smoke",
        help="run the deterministic fictional-data mechanics check",
    )
    smoke.add_argument(
        "--output-dir",
        default=DEFAULT_SMOKE_DIR,
        help="directory for the smoke report and bundle",
    )
    smoke.set_defaults(handler=run_smoke)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``nfl-showcase`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
