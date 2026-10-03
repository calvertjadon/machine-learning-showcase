"""Linear regression with three solvers: closed form, scikit-learn, gradient descent.

This notebook-style script uses percent-format cells, marked with ``# %%``, and
answers one small question. On a noisy linear relationship that the script
generates itself, how closely do three ordinary least squares solvers agree,
and how close does an original batch gradient descent implementation get to the
same solution?

What the script compares
------------------------
* the closed-form least squares solution, computed with ``numpy.linalg.lstsq``;
* :class:`sklearn.linear_model.LinearRegression`;
* :class:`BatchGradientDescentRegressor`, implemented in this file, which
  standardizes features with training-split statistics only and records the
  training loss after every epoch so the script can plot convergence.

The script fits every model on the training split only and uses the held-out
split for the reported R2 and MSE numbers alone. The JSON file it writes holds
every metric quoted elsewhere in the repository, and the script needs no
network access.

Authorship: Jadon Calvert directed this original code, written for the 2026
refresh of the project with AI assistance. Nothing comes from the 2024
coursework, and no course handouts, notebooks or third-party datasets are
reproduced here.

Run with::

    uv run python examples/linear_regression.py --output-dir results/examples
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# The script writes every figure to a PNG file and may run in a terminal, in
# CI or on a machine without a display, so it selects the headless Agg backend.
plt.switch_backend("Agg")

DEFAULT_SEED = 42
N_TRAIN = 200
N_TEST = 100
N_FEATURES = 3
NOISE_STD = 1.0
TRUE_COEFFICIENTS = np.array([3.0, -2.0, 0.5])
TRUE_INTERCEPT = 1.5
GD_LEARNING_RATES = (0.05, 0.1, 0.25)
GD_EPOCHS = 400
PRIMARY_LEARNING_RATE = 0.1


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


# %% Synthetic data
@dataclass(frozen=True)
class LinearDataset:
    """Synthetic regression data with one fixed train/test split."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    coefficients: np.ndarray
    intercept: float


def make_linear_dataset(seed: int) -> LinearDataset:
    """Draw one noisy linear relationship and split it into train and test."""
    rng = np.random.default_rng(seed)
    n_samples = N_TRAIN + N_TEST
    x = rng.standard_normal((n_samples, N_FEATURES))
    signal = x @ TRUE_COEFFICIENTS + TRUE_INTERCEPT
    y = signal + rng.normal(0.0, NOISE_STD, size=n_samples)
    return LinearDataset(
        x_train=x[:N_TRAIN],
        y_train=y[:N_TRAIN],
        x_test=x[N_TRAIN:],
        y_test=y[N_TRAIN:],
        coefficients=TRUE_COEFFICIENTS.copy(),
        intercept=TRUE_INTERCEPT,
    )


# %% Closed-form least squares
def fit_analytical_least_squares(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return ``(intercept, coefficients)`` from the closed-form solution.

    The script solves the normal equations with ``numpy.linalg.lstsq``, which
    uses a singular value decomposition instead of inverting ``X.T @ X``
    directly.
    """
    design = np.column_stack([np.ones(x.shape[0]), x])
    solution = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(solution[0]), solution[1:]


# %% Gradient descent from scratch
class BatchGradientDescentRegressor:
    """Ordinary least squares fitted by batch gradient descent, written here.

    The class standardizes features with the mean and standard deviation of the
    training split only and reuses those statistics at prediction time. It
    records the training mean squared error after each epoch so the script can
    plot convergence.
    """

    def __init__(self, *, learning_rate: float = 0.1, epochs: int = 400) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history_: list[float] = []
        self.coef_scaled_: np.ndarray | None = None
        self.intercept_scaled_ = 0.0
        self.scaler_mean_: np.ndarray | None = None
        self.scaler_scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> BatchGradientDescentRegressor:
        """Fit the model on the training split and return ``self``."""
        self.scaler_mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.scaler_scale_ = scale
        x_scaled = (x - self.scaler_mean_) / self.scaler_scale_
        n_samples, n_features = x_scaled.shape

        coef = np.zeros(n_features)
        intercept = 0.0
        self.loss_history_ = [self._loss(x_scaled, y, coef, intercept)]
        for _ in range(self.epochs):
            residual = x_scaled @ coef + intercept - y
            gradient_coef = (2.0 / n_samples) * (x_scaled.T @ residual)
            gradient_intercept = (2.0 / n_samples) * float(residual.sum())
            coef = coef - self.learning_rate * gradient_coef
            intercept = intercept - self.learning_rate * gradient_intercept
            self.loss_history_.append(self._loss(x_scaled, y, coef, intercept))

        self.coef_scaled_ = coef
        self.intercept_scaled_ = float(intercept)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict targets using the training-split scaling statistics."""
        if self.scaler_mean_ is None or self.scaler_scale_ is None:
            raise NotFittedError("fit() must be called before predict().")
        if self.coef_scaled_ is None:
            raise NotFittedError("fit() must be called before predict().")
        x_scaled = (x - self.scaler_mean_) / self.scaler_scale_
        return x_scaled @ self.coef_scaled_ + self.intercept_scaled_

    @property
    def coefficients_(self) -> np.ndarray:
        """Coefficients expressed in the original, unscaled feature units."""
        if self.coef_scaled_ is None or self.scaler_scale_ is None:
            raise NotFittedError("fit() must be called before coefficients_.")
        return self.coef_scaled_ / self.scaler_scale_

    @property
    def intercept_(self) -> float:
        """Intercept expressed in the original, unscaled feature units."""
        if self.scaler_mean_ is None or self.scaler_scale_ is None:
            raise NotFittedError("fit() must be called before intercept_.")
        if self.coef_scaled_ is None:
            raise NotFittedError("fit() must be called before intercept_.")
        scaled = self.coef_scaled_ * self.scaler_mean_ / self.scaler_scale_
        return self.intercept_scaled_ - float(np.sum(scaled))

    @staticmethod
    def _loss(
        x_scaled: np.ndarray,
        y: np.ndarray,
        coef: np.ndarray,
        intercept: float,
    ) -> float:
        """Mean squared error of the current parameters."""
        residual = x_scaled @ coef + intercept - y
        return float(np.mean(residual**2))


# %% Evaluation helpers
def estimator_metrics(
    dataset: LinearDataset,
    *,
    coefficients: np.ndarray,
    intercept: float,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
) -> dict[str, Any]:
    """Summarize one fitted estimator on the training and held-out splits."""
    coefficients = np.asarray(coefficients)
    return {
        "coefficients": [float(value) for value in coefficients],
        "intercept": float(intercept),
        "train_mse": float(mean_squared_error(dataset.y_train, train_predictions)),
        "test_mse": float(mean_squared_error(dataset.y_test, test_predictions)),
        "test_r2": float(r2_score(dataset.y_test, test_predictions)),
        "max_abs_coefficient_error_vs_truth": float(
            np.max(np.abs(coefficients - dataset.coefficients)),
        ),
        "abs_intercept_error_vs_truth": float(abs(intercept - dataset.intercept)),
    }


# %% Figures
def plot_convergence(
    histories: dict[str, list[float]],
    analytical_train_mse: float,
    output_path: Path,
) -> None:
    """Plot training loss per epoch for each gradient descent learning rate."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for label, history in histories.items():
        ax.plot(np.arange(len(history)), history, label=f"learning rate {label}")
    ax.axhline(
        analytical_train_mse,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="analytical train MSE",
    )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train mean squared error, log scale")
    ax.set_title("Batch gradient descent converges to the least squares optimum")
    ax.legend()
    save_figure(fig, output_path)


def plot_coefficients(
    dataset: LinearDataset,
    estimates: dict[str, tuple[np.ndarray, float]],
    output_path: Path,
) -> None:
    """Compare estimated parameters against the generator's true parameters."""
    truth = np.concatenate([dataset.coefficients, [dataset.intercept]])
    names = [f"w{index}" for index in range(dataset.coefficients.size)]
    names.append("intercept")
    n_series = len(estimates) + 1
    width = 0.8 / n_series
    positions = np.arange(truth.size, dtype=float)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(
        positions + offsets[0],
        truth,
        width=width,
        label="generator truth",
        color="black",
    )
    for index, (name, estimate) in enumerate(estimates.items(), start=1):
        coefficients, intercept = estimate
        values = np.concatenate([coefficients, [intercept]])
        ax.bar(
            positions + offsets[index],
            values,
            width=width,
            label=name,
        )
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xticks(positions, names)
    ax.set_ylabel("parameter value")
    ax.set_title("Estimated parameters after fitting on the training split only")
    ax.legend(fontsize=8)
    save_figure(fig, output_path)


def plot_holdout(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    reports: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot held-out predictions against targets and the residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    combined = np.concatenate([y_test, *predictions.values()])
    lower = float(combined.min())
    upper = float(combined.max())

    for name, values in predictions.items():
        axes[0].scatter(y_test, values, s=18, alpha=0.55, label=name)
        axes[1].scatter(y_test, values - y_test, s=18, alpha=0.55, label=name)
    axes[0].plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linewidth=1.0,
        label="identity",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.0)

    lines = [
        f"{name}: R2 {report['test_r2']:.3f} | MSE {report['test_mse']:.3f}"
        for name, report in reports.items()
    ]
    axes[0].text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    axes[0].set_xlabel("held-out target")
    axes[0].set_ylabel("prediction")
    axes[0].set_title("Held-out predictions")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].set_xlabel("held-out target")
    axes[1].set_ylabel("prediction - target")
    axes[1].set_title("Held-out residuals")
    axes[1].legend(loc="lower right", fontsize=8)
    save_figure(fig, output_path)


# %% Entry point
def main(argv: list[str] | None = None) -> int:
    """Fit all three solvers, write the metrics JSON and the figures."""
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
        help="seed for synthetic data generation",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_linear_dataset(args.seed)

    analytical_intercept, analytical_coef = fit_analytical_least_squares(
        dataset.x_train,
        dataset.y_train,
    )
    sklearn_model = LinearRegression()
    sklearn_model.fit(dataset.x_train, dataset.y_train)
    sklearn_intercept = float(sklearn_model.intercept_)

    gd_models: dict[str, BatchGradientDescentRegressor] = {}
    for learning_rate in GD_LEARNING_RATES:
        label = f"{learning_rate:g}"
        gd_models[label] = BatchGradientDescentRegressor(
            learning_rate=learning_rate,
            epochs=GD_EPOCHS,
        ).fit(dataset.x_train, dataset.y_train)
    primary_gd = gd_models[f"{PRIMARY_LEARNING_RATE:g}"]

    analytic_train = dataset.x_train @ analytical_coef + analytical_intercept
    analytic_predictions = dataset.x_test @ analytical_coef + analytical_intercept
    sklearn_train = sklearn_model.predict(dataset.x_train)
    sklearn_predictions = sklearn_model.predict(dataset.x_test)
    gd_train = primary_gd.predict(dataset.x_train)
    gd_predictions = primary_gd.predict(dataset.x_test)
    predictions = {
        "analytical least squares": analytic_predictions,
        "sklearn LinearRegression": sklearn_predictions,
        "batch gradient descent": gd_predictions,
    }

    reports = {
        "analytical least squares": estimator_metrics(
            dataset,
            coefficients=analytical_coef,
            intercept=analytical_intercept,
            train_predictions=analytic_train,
            test_predictions=analytic_predictions,
        ),
        "sklearn LinearRegression": estimator_metrics(
            dataset,
            coefficients=sklearn_model.coef_,
            intercept=sklearn_intercept,
            train_predictions=sklearn_train,
            test_predictions=sklearn_predictions,
        ),
        "batch gradient descent": estimator_metrics(
            dataset,
            coefficients=primary_gd.coefficients_,
            intercept=primary_gd.intercept_,
            train_predictions=gd_train,
            test_predictions=gd_predictions,
        ),
    }
    reports["batch gradient descent"]["feature_scaling"] = (
        "the mean and standard deviation of the training split, reused at prediction time"
    )

    convergence: dict[str, Any] = {}
    for label, model in gd_models.items():
        convergence[label] = {
            "learning_rate": float(label),
            "epochs": GD_EPOCHS,
            "initial_train_mse": model.loss_history_[0],
            "final_train_mse": model.loss_history_[-1],
            "loss_history": model.loss_history_,
        }

    agreement = {
        "analytical_vs_sklearn_max_abs_coefficient_difference": float(
            np.max(np.abs(analytical_coef - sklearn_model.coef_)),
        ),
        "analytical_vs_gradient_descent_max_abs_coefficient_difference": float(
            np.max(np.abs(analytical_coef - primary_gd.coefficients_)),
        ),
    }

    payload = {
        "example": "linear_regression",
        "author": "Jadon Calvert",
        "generated_by": "examples/linear_regression.py",
        "seed": args.seed,
        "environment": environment_info(),
        "protocol": (
            "The script fits every model on the training split only and uses "
            "the held-out split for the reported R2 and MSE values alone."
        ),
        "data": {
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "n_features": N_FEATURES,
            "noise_std": NOISE_STD,
            "true_coefficients": [float(value) for value in dataset.coefficients],
            "true_intercept": float(dataset.intercept),
        },
        "estimators": reports,
        "convergence": convergence,
        "agreement": agreement,
    }

    metrics_path = args.output_dir / "linear_regression_metrics.json"
    write_json(metrics_path, payload)

    histories = {label: model.loss_history_ for label, model in gd_models.items()}
    plot_convergence(
        histories,
        reports["analytical least squares"]["train_mse"],
        args.output_dir / "linear_regression_convergence.png",
    )
    estimates = {
        "analytical least squares": (analytical_coef, analytical_intercept),
        "sklearn LinearRegression": (sklearn_model.coef_, sklearn_intercept),
        "batch gradient descent": (primary_gd.coefficients_, primary_gd.intercept_),
    }
    plot_coefficients(
        dataset,
        estimates,
        args.output_dir / "linear_regression_coefficients.png",
    )
    plot_holdout(
        dataset.y_test,
        predictions,
        reports,
        args.output_dir / "linear_regression_holdout.png",
    )

    print(f"== linear regression example, seed {args.seed} ==")
    print(f"train rows: {N_TRAIN} | test rows: {N_TEST} | features: {N_FEATURES}")
    for name, report in reports.items():
        print(f"{name}:")
        print(f"  train MSE {report['train_mse']:.4f}")
        print(f"  test MSE {report['test_mse']:.4f} | test R2 {report['test_r2']:.4f}")
    gd_report = reports["batch gradient descent"]
    print(f"gradient descent final train MSE {gd_report['train_mse']:.6f}")
    print(f"metrics written to {metrics_path}")
    print(f"figures written to {args.output_dir}/linear_regression_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
