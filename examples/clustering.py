"""Unsupervised grouping: k-means on generated blobs and image color quantization.

This notebook-style script uses percent-format cells, marked with ``# %%``, and
runs two unsupervised studies. Both studies generate their data inside the
script.

1. Blobs. The script draws a seeded two-dimensional mixture with four well
   separated centers and clusters it with k-means for every k in a small range.
   It records inertia and silhouette for each k, compares the partition at the
   generator's k with the true labels using the adjusted Rand index, and plots
   the diagnostics so the elbow and the silhouette peak can be inspected.
2. Image quantization. The script renders an original synthetic RGB scene in
   numpy and approximates it with a k-means palette. It reports the
   quantization error as RMSE, MAE and PSNR, the fraction of pixels whose
   color changed, and the compression arithmetic for an indexed palette. No
   photograph, course image or downloaded asset appears anywhere.

The script writes a JSON file that holds every metric quoted elsewhere in the
repository, and it needs no network access.

Authorship: Jadon Calvert directed this original code, written for the 2026
refresh of the project with AI assistance. Nothing comes from the 2024
coursework, and no course handouts, notebooks or third-party images are
reproduced here.

Run with::

    uv run python examples/clustering.py --output-dir results/examples
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
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

# The script writes every figure to a PNG file and may run in a terminal, in
# CI or on a machine without a display, so it selects the headless Agg backend.
plt.switch_backend("Agg")

DEFAULT_SEED = 42
BLOB_SAMPLES = 600
BLOB_STD = 0.65
BLOB_CENTERS = np.array(
    [
        [0.0, 0.0],
        [6.0, 6.0],
        [-6.0, 4.5],
        [4.0, -6.0],
    ],
)
K_RANGE = tuple(range(2, 9))
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 288
IMAGE_COLORS = 8
IMAGE_NOISE_STD = 4.0


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


# %% Synthetic blobs
@dataclass(frozen=True)
class BlobStudy:
    """Blob sample, fitted partition and JSON-ready summary."""

    features: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    true_centers: np.ndarray
    summary: dict[str, Any]


def run_blob_study(seed: int) -> BlobStudy:
    """Cluster seeded blobs across a range of k and summarize the sweep."""
    features, truth = make_blobs(
        n_samples=BLOB_SAMPLES,
        centers=BLOB_CENTERS,
        n_features=2,
        cluster_std=BLOB_STD,
        random_state=seed,
    )

    sweep: list[dict[str, Any]] = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        model.fit(features)
        sweep.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(features, model.labels_)),
            },
        )

    selected_k = int(BLOB_CENTERS.shape[0])
    selected = KMeans(n_clusters=selected_k, n_init=10, random_state=seed)
    selected.fit(features)
    sizes = np.bincount(selected.labels_, minlength=selected_k)
    best_by_silhouette = max(sweep, key=lambda entry: entry["silhouette"])

    summary = {
        "n_samples": BLOB_SAMPLES,
        "cluster_std": BLOB_STD,
        "true_centers": [[float(value) for value in row] for row in BLOB_CENTERS],
        "k_sweep": sweep,
        "selected_k": selected_k,
        "selected": {
            "inertia": float(selected.inertia_),
            "silhouette": float(silhouette_score(features, selected.labels_)),
            "adjusted_rand_index": float(
                adjusted_rand_score(truth, selected.labels_),
            ),
            "iterations": int(selected.n_iter_),
            "cluster_sizes": [int(value) for value in sizes],
            "centroids": [[float(value) for value in row] for row in selected.cluster_centers_],
        },
        "best_k_by_silhouette": int(best_by_silhouette["k"]),
    }
    return BlobStudy(
        features=features,
        labels=selected.labels_,
        centroids=selected.cluster_centers_,
        true_centers=BLOB_CENTERS,
        summary=summary,
    )


# %% Synthetic image
def make_synthetic_scene(seed: int) -> np.ndarray:
    """Render an original RGB scene with gradients, texture and shapes."""
    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    x = columns / (IMAGE_WIDTH - 1)
    y = rows / (IMAGE_HEIGHT - 1)

    red = 30.0 + 190.0 * x
    green = 40.0 + 180.0 * y
    blue = 70.0 + 130.0 * (1.0 - x) * (1.0 - y)
    texture = 30.0 * np.sin(6.0 * np.pi * x) * np.cos(4.0 * np.pi * y)
    red = red + texture
    blue = blue - 0.5 * texture

    circle_x = 0.28 * IMAGE_WIDTH
    circle_y = 0.30 * IMAGE_HEIGHT
    circle_radius = 0.17 * IMAGE_HEIGHT
    distance = (columns - circle_x) ** 2 + (rows - circle_y) ** 2
    circle = distance <= circle_radius**2
    red = np.where(circle, 245.0, red)
    green = np.where(circle, 200.0, green)
    blue = np.where(circle, 45.0, blue)

    rectangle = (rows >= 0.55 * IMAGE_HEIGHT) & (rows <= 0.88 * IMAGE_HEIGHT)
    rectangle = rectangle & (columns >= 0.06 * IMAGE_WIDTH)
    rectangle = rectangle & (columns <= 0.34 * IMAGE_WIDTH)
    red = np.where(rectangle, 30.0, red)
    green = np.where(rectangle, 170.0, green)
    blue = np.where(rectangle, 165.0, blue)

    triangle_x = (columns - 0.62 * IMAGE_WIDTH) / (0.16 * IMAGE_WIDTH)
    triangle_y = (rows - 0.06 * IMAGE_HEIGHT) / (0.42 * IMAGE_HEIGHT)
    triangle = (triangle_y >= 0.0) & (triangle_y <= 1.0)
    triangle = triangle & (np.abs(triangle_x) <= 1.0 - triangle_y)
    red = np.where(triangle, 205.0, red)
    green = np.where(triangle, 45.0, green)
    blue = np.where(triangle, 170.0, blue)

    diagonal = np.abs(x - y) < 0.045
    red = np.where(diagonal, 250.0, red)
    green = np.where(diagonal, 250.0, green)
    blue = np.where(diagonal, 235.0, blue)

    scene = np.stack([red, green, blue], axis=-1)
    scene = scene + rng.normal(0.0, IMAGE_NOISE_STD, size=scene.shape)
    return np.clip(scene, 0.0, 255.0).astype(np.uint8)


@dataclass(frozen=True)
class ImageStudy:
    """Synthetic scene, its quantized version, palette and JSON summary."""

    image: np.ndarray
    segmented: np.ndarray
    palette: np.ndarray
    summary: dict[str, Any]


def run_image_study(seed: int) -> ImageStudy:
    """Quantize an original synthetic scene with a k-means color palette."""
    image = make_synthetic_scene(seed)
    height, width = image.shape[0], image.shape[1]
    pixels = image.reshape(-1, 3).astype(np.float64)

    model = KMeans(n_clusters=IMAGE_COLORS, n_init=10, random_state=seed)
    labels = model.fit_predict(pixels)
    palette = np.clip(np.rint(model.cluster_centers_), 0, 255).astype(np.uint8)
    segmented = palette[labels].reshape(height, width, 3)

    difference = segmented.astype(np.float64) - image.astype(np.float64)
    squared = difference**2
    rmse = float(np.sqrt(float(squared.mean())))
    psnr_db = float(20.0 * np.log10(255.0 / rmse)) if rmse > 0.0 else None
    per_channel_rmse = np.sqrt(squared.mean(axis=(0, 1)))

    n_pixels = height * width
    index_bits = int(np.ceil(np.log2(IMAGE_COLORS)))
    index_bytes = (n_pixels * index_bits + 7) // 8
    palette_bytes = IMAGE_COLORS * 3
    source_bytes = int(image.size)
    total_bytes = index_bytes + palette_bytes
    shares = np.bincount(labels, minlength=IMAGE_COLORS) / labels.size

    summary = {
        "height": height,
        "width": width,
        "n_pixels": n_pixels,
        "n_colors": IMAGE_COLORS,
        "noise_std": IMAGE_NOISE_STD,
        "kmeans_iterations": int(model.n_iter_),
        "kmeans_inertia": float(model.inertia_),
        "palette_rgb": [[int(value) for value in row] for row in palette],
        "cluster_pixel_shares": [float(value) for value in shares],
        "quantization": {
            "rmse": rmse,
            "mae": float(np.mean(np.abs(difference))),
            "psnr_db": psnr_db,
            "per_channel_rmse": {
                "red": float(per_channel_rmse[0]),
                "green": float(per_channel_rmse[1]),
                "blue": float(per_channel_rmse[2]),
            },
            "pixels_changed_fraction": float(
                np.mean(np.any(segmented != image, axis=2)),
            ),
            "unique_colors_source": int(
                np.unique(image.reshape(-1, 3), axis=0).shape[0],
            ),
            "unique_colors_quantized": int(
                np.unique(segmented.reshape(-1, 3), axis=0).shape[0],
            ),
        },
        "compression": {
            "source_bits_per_pixel": 24,
            "index_bits_per_pixel": index_bits,
            "index_stream_ratio": float(24 / index_bits),
            "source_bytes": source_bytes,
            "index_bytes": int(index_bytes),
            "palette_bytes": int(palette_bytes),
            "total_bytes_with_palette": int(total_bytes),
            "total_ratio_with_palette": float(source_bytes / total_bytes),
            "note": (
                "The index stream ratio ignores the palette; the total ratio "
                "includes one palette entry per color."
            ),
        },
    }
    return ImageStudy(
        image=image,
        segmented=segmented,
        palette=palette,
        summary=summary,
    )


# %% Figures
def plot_blob_study(study: BlobStudy, output_path: Path) -> None:
    """Plot the selected k-means partition and the k diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].scatter(
        study.features[:, 0],
        study.features[:, 1],
        c=study.labels,
        cmap="tab10",
        s=14,
        alpha=0.8,
    )
    axes[0].scatter(
        study.centroids[:, 0],
        study.centroids[:, 1],
        marker="X",
        s=150,
        color="black",
        label="k-means centroids",
    )
    axes[0].scatter(
        study.true_centers[:, 0],
        study.true_centers[:, 1],
        marker="+",
        s=180,
        color="white",
        linewidths=2.0,
        label="generator centers",
    )
    selected_k = study.summary["selected_k"]
    axes[0].set_title(f"k-means partition with k={selected_k}")
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].legend(loc="best", fontsize=8)

    k_values = [entry["k"] for entry in study.summary["k_sweep"]]
    inertias = [entry["inertia"] for entry in study.summary["k_sweep"]]
    silhouettes = [entry["silhouette"] for entry in study.summary["k_sweep"]]
    axes[1].plot(k_values, inertias, marker="o", label="inertia, left axis")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("inertia")
    axes[1].axvline(
        selected_k,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="generator k",
    )
    twin = axes[1].twinx()
    twin.plot(
        k_values,
        silhouettes,
        marker="s",
        color="tab:orange",
        label="silhouette, right axis",
    )
    twin.set_ylabel("silhouette, right axis")
    axes[1].set_title("K selection diagnostics")
    handles = axes[1].get_lines() + twin.get_lines()
    names = [line.get_label() for line in handles]
    axes[1].legend(handles, names, loc="best", fontsize=8)
    save_figure(fig, output_path)


def plot_image_study(study: ImageStudy, output_path: Path) -> None:
    """Plot the original scene, the quantized image, palette and error map."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    axes[0, 0].imshow(study.image)
    axes[0, 0].set_title("Original synthetic scene")
    axes[0, 1].imshow(study.segmented)
    axes[0, 1].set_title(f"Quantized to {study.palette.shape[0]} colors")
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1, 0].imshow(study.palette.reshape(1, -1, 3), aspect="auto")
    axes[1, 0].set_yticks([])
    shares = study.summary["cluster_pixel_shares"]
    share_labels = [f"{share:.1%}" for share in shares]
    axes[1, 0].set_xticks(np.arange(study.palette.shape[0]), share_labels, rotation=45)
    axes[1, 0].set_title("Palette colors, with the pixel share of each color")

    error = np.abs(study.segmented.astype(np.float64) - study.image.astype(np.float64))
    error_map = error.mean(axis=2)
    artist = axes[1, 1].imshow(error_map, cmap="magma")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title("Mean absolute error per pixel")
    fig.colorbar(artist, ax=axes[1, 1], fraction=0.046, pad=0.04)
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
        help="seed for blob generation, k-means restarts and image noise",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    blob_study = run_blob_study(args.seed)
    image_study = run_image_study(args.seed)

    payload = {
        "example": "clustering",
        "author": "Jadon Calvert",
        "generated_by": "examples/clustering.py",
        "seed": args.seed,
        "environment": environment_info(),
        "blobs": blob_study.summary,
        "image_quantization": image_study.summary,
    }
    metrics_path = args.output_dir / "clustering_metrics.json"
    write_json(metrics_path, payload)

    plot_blob_study(blob_study, args.output_dir / "clustering_blobs.png")
    plot_image_study(
        image_study,
        args.output_dir / "clustering_image_quantization.png",
    )

    blob_summary = blob_study.summary
    selected = blob_summary["selected"]
    print(f"== clustering example, seed {args.seed} ==")
    print(f"blobs: {blob_summary['n_samples']} generated points")
    print(f"  k={blob_summary['selected_k']}: inertia {selected['inertia']:.1f}")
    print(f"  silhouette {selected['silhouette']:.3f}")
    print(f"  adjusted Rand index vs truth {selected['adjusted_rand_index']:.3f}")
    print(f"  best k by silhouette: {blob_summary['best_k_by_silhouette']}")

    image_summary = image_study.summary
    quantization = image_summary["quantization"]
    compression = image_summary["compression"]
    print(
        f"image: {image_summary['height']}x{image_summary['width']} pixels, "
        f"{image_summary['n_colors']} colors",
    )
    psnr = quantization["psnr_db"]
    psnr_text = "undefined, the error is zero" if psnr is None else f"{psnr:.2f} dB"
    print(f"  quantization RMSE {quantization['rmse']:.2f} | PSNR {psnr_text}")
    print(
        f"  pixels changed {quantization['pixels_changed_fraction']:.1%} | "
        f"index {compression['index_bits_per_pixel']} bits/pixel",
    )
    print(f"  source colors {quantization['unique_colors_source']}")
    print(f"metrics written to {metrics_path}")
    print(f"figures written to {args.output_dir}/clustering_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
