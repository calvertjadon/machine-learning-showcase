"""Classifiers on two very different datasets: tabular and toy text.

This notebook-style script (percent-format cells, ``# %%``) runs two compact
classification studies that share a measurement style, not a dataset.

1. Breast cancer (tabular, bundled with scikit-learn): a k-nearest-neighbors
   pipeline and a Gaussian naive Bayes pipeline, both wrapped around a
   StandardScaler that is fitted on the training split only, compared on a
   single stratified holdout. The neighbor count for KNN is chosen by 5-fold
   stratified cross-validation on the training split, never on the holdout.
2. Toy messages (text, written for this repository): a fictional and
   deliberately tiny spam/ham corpus that demonstrates TF-IDF vectorization
   with multinomial naive Bayes. The corpus is original teaching text, not a
   real SMS dataset, and its scores measure pipeline mechanics on a handful of
   sentences rather than real-world spam filtering.

The two studies are never compared with each other. The JSON file written by
this script is the source of truth for every metric quoted elsewhere in the
repository, and no network access is required.

Authorship: original fictional text and code created for this 2026 showcase
refresh with AI assistance under Jadon Calvert's direction; not copied from the
2024 coursework, and no course handouts, real messages or third-party datasets
are reproduced here.

Run with::

    uv run python examples/classification.py --output-dir results/examples
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Figures are always written to PNG files, so pin a headless backend that
# behaves identically in a terminal, in CI runs and in headless renders.
plt.switch_backend("Agg")

DEFAULT_SEED = 42
HOLDOUT_FRACTION = 0.25
CV_FOLDS = 5
NEIGHBOR_GRID = (1, 3, 5, 7, 9, 11, 15)

# The toy corpus below is original fictional teaching text written for this
# repository. It is intentionally tiny, contains no real messages, URLs or
# phone numbers, and must not be treated as a spam-filter benchmark.
TOY_MESSAGES: tuple[tuple[str, str], ...] = (
    ("ham", "Standup moved to 9:30 in room 4, the invite has the details."),
    ("ham", "Can you send the draft agenda before lunch? Thanks!"),
    ("ham", "Your parcel arrives today between 14:00 and 16:00."),
    ("ham", "Reminder: the library books are due back on Friday."),
    ("ham", "Dinner at the usual place at seven? Let me know."),
    ("ham", "The lab printer is out of paper, facilities has been notified."),
    ("ham", "Meeting notes are in the shared folder, actions at the top."),
    ("ham", "I found your umbrella in the break room, it is at reception."),
    ("ham", "The train is ten minutes late, I will still be there by six."),
    ("ham", "Practice is cancelled tonight because the gym is closed."),
    ("ham", "Could you review my pull request when you have a moment?"),
    ("ham", "Happy birthday! Let us celebrate this weekend."),
    ("ham", "The signed invoice copy is in the shared drive now."),
    ("ham", "Do you need anything from the shop on my way home?"),
    ("ham", "The workshop moved to Thursday afternoon, same room."),
    ("ham", "Thanks for the feedback on the draft, I updated the file."),
    ("spam", "WIN a luxury cruise! Reply PRIZE now to claim your free cabin."),
    ("spam", "URGENT: your account closes today unless you verify it now."),
    ("spam", "Congratulations! You are selected for a cash reward, call now."),
    ("spam", "FREE ringtones for new subscribers, text YES to join today."),
    ("spam", "Claim your free gift card by clicking this link right now."),
    ("spam", "Lowest mortgage rates in town, no credit check, apply today."),
    ("spam", "You are a winner! Send your details to receive the prize."),
    ("spam", "Limited offer: buy one phone and get three free, act fast."),
    ("spam", "Your loan is approved, reply with your bank details now."),
    ("spam", "Exclusive deal for the first hundred callers, dial today."),
    ("spam", "Make money fast from home with no experience, join free."),
    ("spam", "Your refund is waiting, confirm your card number right now."),
    ("spam", "Hot discount codes inside, click the link before midnight."),
    ("spam", "Final notice: claim your unclaimed lottery winnings today."),
    ("spam", "Free entry to the grand draw, text DRAW to the number below."),
    ("spam", "Cheap meds online with no prescription, order right now."),
)


# %% Utilities
def json_ready(value: Any) -> Any:
    """Convert numpy and path objects into plain JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a metrics payload as sorted, indented UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a matplotlib figure as a PNG and release it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def environment_info() -> dict[str, str]:
    """Record the library versions that produced the measurements."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": mpl.__version__,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> dict[str, Any]:
    """Return accuracy, macro F1, a per-class report and a confusion matrix."""
    labels = list(range(len(target_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0),
        ),
        "labels": target_names,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
    }


# %% Breast cancer study
def build_knn_pipeline() -> Pipeline:
    """Return a KNN pipeline that scales features before voting."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ],
    )


def select_knn_neighbors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[GridSearchCV, dict[str, Any]]:
    """Choose k with 5-fold stratified cross-validation on the training split."""
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    search = GridSearchCV(
        estimator=build_knn_pipeline(),
        param_grid={"knn__n_neighbors": list(NEIGHBOR_GRID)},
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train)
    summary = {
        "grid": [int(value) for value in NEIGHBOR_GRID],
        "mean_cv_macro_f1": [float(value) for value in search.cv_results_["mean_test_score"]],
        "std_cv_macro_f1": [float(value) for value in search.cv_results_["std_test_score"]],
        "best_k": int(search.best_params_["knn__n_neighbors"]),
        "best_mean_cv_macro_f1": float(search.best_score_),
        "scoring": "macro F1, 5-fold stratified cross-validation on the train split",
    }
    return search, summary


def run_breast_cancer_study(seed: int) -> dict[str, Any]:
    """Compare scaled KNN and GaussianNB on one stratified holdout."""
    bunch = load_breast_cancer()
    features = np.asarray(bunch.data, dtype=np.float64)
    labels = np.asarray(bunch.target, dtype=np.int64)
    target_names = [str(name) for name in bunch.target_names]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=HOLDOUT_FRACTION,
        stratify=labels,
        random_state=seed,
    )

    search, knn_cv = select_knn_neighbors(x_train, y_train, seed)
    knn_model = search.best_estimator_
    gaussian_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("nb", GaussianNB()),
        ],
    )
    gaussian_model.fit(x_train, y_train)

    return {
        "source": "sklearn.datasets.load_breast_cancer (bundled, no download)",
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "class_names": target_names,
        "holdout": {
            "test_fraction": HOLDOUT_FRACTION,
            "stratified": True,
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
        },
        "knn": {
            "cv_selection": knn_cv,
            **classification_metrics(
                y_test,
                knn_model.predict(x_test),
                target_names,
            ),
        },
        "gaussian_nb": classification_metrics(
            y_test,
            gaussian_model.predict(x_test),
            target_names,
        ),
        "notes": (
            "A single stratified split is reported; the scaler and KNN's k are "
            "fitted on the training split only. These scores describe this "
            "split, not the dataset as a whole, and GaussianNB is scale "
            "invariant, so its scaler only keeps the pipeline convention."
        ),
    }


# %% Toy text study
def build_text_pipeline() -> Pipeline:
    """Return the TF-IDF plus multinomial naive Bayes teaching pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("nb", MultinomialNB(alpha=1.0)),
        ],
    )


def run_toy_text_study(seed: int) -> dict[str, Any]:
    """Fit TF-IDF plus MultinomialNB on the fictional toy message corpus."""
    messages = np.array([text for _, text in TOY_MESSAGES], dtype=str)
    labels = np.array([label for label, _ in TOY_MESSAGES], dtype=str)
    target_names = ["ham", "spam"]
    numeric_labels = (labels == "spam").astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        messages,
        numeric_labels,
        test_size=HOLDOUT_FRACTION,
        stratify=numeric_labels,
        random_state=seed,
    )
    holdout_model = build_text_pipeline()
    holdout_model.fit(x_train, y_train)
    holdout_metrics = classification_metrics(
        y_test,
        holdout_model.predict(x_test),
        target_names,
    )
    holdout_metrics["vocabulary_size_train"] = int(
        len(holdout_model.named_steps["tfidf"].vocabulary_),
    )

    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=seed,
    )
    fold_scores = cross_val_score(
        build_text_pipeline(),
        messages,
        numeric_labels,
        cv=cross_validation,
        scoring="f1_macro",
        n_jobs=1,
    )

    # The token indicators reuse the training-only holdout model: no refit on
    # the full toy corpus, so the descriptive vocabulary comes from the same
    # training split as the reported holdout scores.
    vocabulary = holdout_model.named_steps["tfidf"].get_feature_names_out()
    log_probabilities = holdout_model.named_steps["nb"].feature_log_prob_
    # MultinomialNB sorts classes, so index 0 is ham and index 1 is spam here.
    spam_minus_ham = log_probabilities[1] - log_probabilities[0]
    order = np.argsort(spam_minus_ham)
    most_ham = [str(vocabulary[index]) for index in order[:8]]
    most_spam = [str(vocabulary[index]) for index in order[-8:]][::-1]

    return {
        "corpus": (
            "Original fictional teaching messages written for this repository; "
            "not a real SMS dataset and not a benchmark of any kind."
        ),
        "n_messages": len(TOY_MESSAGES),
        "class_counts": {
            "ham": int(np.sum(numeric_labels == 0)),
            "spam": int(np.sum(numeric_labels == 1)),
        },
        "pipeline": (
            "TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True) -> MultinomialNB(alpha=1.0)"
        ),
        "holdout": {
            "test_fraction": HOLDOUT_FRACTION,
            "stratified": True,
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            **holdout_metrics,
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "macro_f1_per_fold": [float(value) for value in fold_scores],
            "macro_f1_mean": float(fold_scores.mean()),
            "macro_f1_std": float(fold_scores.std()),
            "vocabulary_fit": "inside each fold through the pipeline (train-only)",
        },
        "token_indicators": {
            "fit_on": (
                "training split only (the same pipeline fit that produced the "
                "holdout metrics; no full-corpus refit)"
            ),
            "vocabulary_size": int(vocabulary.size),
            "most_spam_associated": most_spam,
            "most_ham_associated": most_ham,
        },
        "limitations": (
            "About thirty fictional sentences; any high score reflects this toy "
            "corpus, not real spam-filter quality, and the corpus is far too "
            "small for generalization claims."
        ),
    }


# %% Figures
def plot_confusion_pair(
    matrices: dict[str, np.ndarray],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Plot one annotated confusion matrix per model, side by side."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(11.0, 4.6))
    for ax, (name, matrix) in zip(axes, matrices.items(), strict=True):
        artist = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"{name} on the breast cancer holdout")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(len(class_names)), class_names)
        ax.set_yticks(range(len(class_names)), class_names)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    str(int(matrix[row, column])),
                    ha="center",
                    va="center",
                )
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_path)


def plot_model_comparison(
    metrics: dict[str, dict[str, Any]],
    cv_summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot holdout bars and the KNN cross-validation selection curve."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    names = list(metrics)
    positions = np.arange(len(names), dtype=float)
    width = 0.35
    accuracy = [metrics[name]["accuracy"] for name in names]
    macro_f1 = [metrics[name]["macro_f1"] for name in names]
    axes[0].bar(positions - width / 2, accuracy, width, label="accuracy")
    axes[0].bar(positions + width / 2, macro_f1, width, label="macro F1")
    axes[0].set_xticks(positions, names)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Breast cancer holdout scores")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].errorbar(
        cv_summary["grid"],
        cv_summary["mean_cv_macro_f1"],
        yerr=cv_summary["std_cv_macro_f1"],
        marker="o",
        capsize=3,
    )
    axes[1].set_xlabel("k (number of neighbors)")
    axes[1].set_ylabel("macro F1 (5-fold CV on train)")
    axes[1].set_title("KNN neighbor count selected on the training split")
    save_figure(fig, output_path)


def plot_toy_text(study: dict[str, Any], output_path: Path) -> None:
    """Plot the toy-corpus holdout confusion matrix and per-fold CV scores."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    matrix = np.asarray(study["holdout"]["confusion_matrix"])
    class_names = study["holdout"]["labels"]
    artist = axes[0].imshow(matrix, cmap="Purples")
    axes[0].set_title("Toy corpus: stratified holdout")
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("true")
    axes[0].set_xticks(range(len(class_names)), class_names)
    axes[0].set_yticks(range(len(class_names)), class_names)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[0].text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
            )
    fig.colorbar(artist, ax=axes[0], fraction=0.046, pad=0.04)

    fold_scores = study["cross_validation"]["macro_f1_per_fold"]
    axes[1].bar(np.arange(len(fold_scores)), fold_scores, color="tab:purple")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("fold")
    axes[1].set_ylabel("macro F1")
    mean_score = study["cross_validation"]["macro_f1_mean"]
    axes[1].set_title(f"5-fold CV on the toy corpus (mean {mean_score:.3f})")
    save_figure(fig, output_path)


# %% Entry point
def main(argv: list[str] | None = None) -> int:
    """Run both studies, write the metrics JSON and the figures."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/examples"),
        help="directory that receives the metrics JSON and figures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seed for the stratified splits and cross-validation folds",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    breast_cancer = run_breast_cancer_study(args.seed)
    toy_text = run_toy_text_study(args.seed)

    payload = {
        "example": "classification",
        "author": "Jadon Calvert",
        "generated_by": "examples/classification.py",
        "seed": args.seed,
        "environment": environment_info(),
        "breast_cancer": breast_cancer,
        "toy_text": toy_text,
    }
    metrics_path = args.output_dir / "classification_metrics.json"
    write_json(metrics_path, payload)

    knn_metrics = breast_cancer["knn"]
    gaussian_metrics = breast_cancer["gaussian_nb"]
    confusion_matrices = {
        "KNN": np.asarray(knn_metrics["confusion_matrix"]),
        "GaussianNB": np.asarray(gaussian_metrics["confusion_matrix"]),
    }
    plot_confusion_pair(
        confusion_matrices,
        breast_cancer["class_names"],
        args.output_dir / "classification_breast_cancer_confusion.png",
    )
    plot_model_comparison(
        {"KNN": knn_metrics, "GaussianNB": gaussian_metrics},
        knn_metrics["cv_selection"],
        args.output_dir / "classification_breast_cancer_comparison.png",
    )
    plot_toy_text(toy_text, args.output_dir / "classification_toy_text.png")

    print(f"== classification example (seed {args.seed}) ==")
    print(f"breast cancer samples: {breast_cancer['n_samples']}")
    print(f"  holdout rows: {breast_cancer['holdout']['n_test']}")
    knn_accuracy = knn_metrics["accuracy"]
    knn_f1 = knn_metrics["macro_f1"]
    nb_accuracy = gaussian_metrics["accuracy"]
    nb_f1 = gaussian_metrics["macro_f1"]
    print(f"  KNN: accuracy {knn_accuracy:.3f} | macro F1 {knn_f1:.3f}")
    print(f"  GaussianNB: accuracy {nb_accuracy:.3f} | macro F1 {nb_f1:.3f}")
    print(f"  KNN best k: {knn_metrics['cv_selection']['best_k']}")
    toy_cv = toy_text["cross_validation"]
    holdout_f1 = toy_text["holdout"]["macro_f1"]
    print(f"toy corpus: {toy_text['n_messages']} fictional messages")
    print(f"  holdout macro F1 {holdout_f1:.3f}")
    print(f"  5-fold CV macro F1 mean {toy_cv['macro_f1_mean']:.3f}")
    print(f"  fold scores {toy_cv['macro_f1_per_fold']}")
    print(f"metrics -> {metrics_path}")
    print(f"figures -> {args.output_dir}/classification_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
