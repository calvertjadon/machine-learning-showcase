# Curated examples

Three self-contained, notebook-style scripts that each build a small machine
learning question from scratch, measure it, and write the evidence as JSON plus
original figures. These are original implementations created for this
showcase (2026) and developed with AI assistance under Jadon Calvert's
direction; they are not copies of the 2024 coursework notebooks and do not
claim that every line was personally hand-authored. They do not import the
`nfl_showcase` package, do not read course material, and do not touch the
network: every dataset is either bundled with scikit-learn or generated inside
the script.

The `# %%` markers follow the percent (Jupytext) notebook convention, so each
file opens as a sequence of cells in a notebook environment and still runs
directly as a normal Python script.

## Questions and approach

### `linear_regression.py`

Question: on a noisy linear relationship that we generate ourselves, how
closely do the closed-form least squares solution, scikit-learn's
`LinearRegression`, and an original batch gradient descent implementation agree
on the same training split?

- Data: three standard-normal features, known coefficients, Gaussian noise,
  200 training rows and 100 held-out rows, all from one seed.
- Closed-form fit via `numpy.linalg.lstsq`; the sklearn estimator is fitted for
  comparison; parameter differences between them are reported.
- The gradient descent regressor standardizes features with training-split
  statistics only, records its training loss every epoch, and is run at several
  learning rates. Convergence is plotted on a log scale against the analytical
  training MSE.
- Reported metrics include true-vs-estimated parameters, training and held-out
  MSE, and held-out R² per solver.

### `classification.py`

Two independent studies that share a measurement style but never compete with
each other.

Breast cancer (tabular): a `StandardScaler -> KNeighborsClassifier` pipeline and
a `StandardScaler -> GaussianNB` pipeline on the dataset bundled with
scikit-learn. The split is stratified; the scaler and the neighbor count are
fitted on the training split only, with `k` selected by 5-fold stratified
cross-validation. Reported metrics: accuracy, macro F1, per-class
classification report, and the confusion matrix for each model.

Toy messages (text): TF-IDF (unigrams and bigrams, sublinear term frequency)
followed by `MultinomialNB`, applied to a small spam/ham corpus that was written
for this repository and is explicitly fictional. A stratified holdout and 5-fold
cross-validation are reported, together with the tokens the fitted model
associates most strongly with each class. Those indicator tokens (and their
vocabulary size) come from the same training-split fit that produced the holdout
metrics, and each cross-validation fold refits within its training folds; no
model is fitted on the full toy corpus.

### `clustering.py`

Blobs: a seeded four-center mixture. k-means is fitted for every `k` in a small
range and inertia plus silhouette are recorded per `k`; the partition at the
generator's `k` is compared with the true labels using the adjusted Rand index.

Image quantization: an original synthetic RGB scene (gradients, texture,
geometric shapes, seeded noise) is rendered in numpy and approximated with a
k-means palette. The script reports the quantization error (RMSE, MAE, PSNR,
per-channel RMSE, fraction of pixels changed) and the compression arithmetic for
an indexed palette. No photograph or third-party image is used.

## Measured results

Every number below was checked against the JSON files written by the recorded
`--seed 42` runs under `results/examples/`. This section is a reviewed static
summary, not prose generated from the JSON at build time; if the outputs are
regenerated (different seed or library versions), re-check the numbers here
before relying on them.

- **`linear_regression`** (`linear_regression_metrics.json`): all three
  solvers recover the same line on the 100 held-out rows — test R² 0.9213 and
  test MSE 1.0703 each (train MSE 0.8949). The estimated coefficients
  `[2.9521, -1.9027, 0.4191]` sit near the true `[3.0, -2.0, 0.5]` (max
  absolute coefficient error 0.0973), and the pairwise maximum coefficient
  differences between the analytical solution, scikit-learn, and batch
  gradient descent are floating-point noise (2.1e-15 and 2.4e-15). At learning
  rate 0.25 the gradient-descent loss falls from 15.3842 to 0.8950 within
  roughly 35 of its 400 epochs and then stays flat.
- **`classification`** (`classification_metrics.json`): breast cancer, 426
  train / 143 stratified held-out rows — KNN with `k=9` (selected by 5-fold
  stratified CV, mean macro-F1 0.9669) reaches 0.9650 holdout accuracy and
  0.9617 macro-F1; Gaussian naive Bayes reaches 0.9371 / 0.9317. Toy text —
  the TF-IDF + MultinomialNB pipeline scores 1.0 accuracy and macro-F1 on its
  8-message stratified holdout, with 5-fold CV macro-F1 0.9003 ± 0.0816; that
  is a toy-corpus score, not a spam-filter benchmark.
- **`clustering`** (`clustering_metrics.json`): blobs — silhouette peaks at
  the generator's `k=4` (0.8476, inertia 492.13) and the recovered partition
  matches the true labels exactly (adjusted Rand index 1.0, four clusters of
  150). Image quantization — the 288×192 synthetic scene (55,296 pixels,
  46,822 unique source colors) is reduced to an 8-color palette with RMSE
  20.30, MAE 15.56, PSNR 21.98 dB; 99.99% of pixels change, and the indexed
  palette stream is 20,760 bytes versus 165,888 source bytes (7.99× including
  the 24-byte palette).

## Commands

Run from the repository root after `uv sync --locked`:

```bash
uv run python examples/linear_regression.py --output-dir results/examples
uv run python examples/classification.py --output-dir results/examples
uv run python examples/clustering.py --output-dir results/examples
```

Each script is independent and can be run alone. `--output-dir` defaults to
`results/examples`, so the flag can be omitted; `--seed` defaults to `42` and
exists only to re-run an example with a different draw. Data sizes are
deliberately small (hundreds of samples and one small raster), so each script
finishes quickly on a laptop CPU.

## Outputs

| Script | Metrics JSON | Figures |
| --- | --- | --- |
| `linear_regression.py` | `linear_regression_metrics.json` | `linear_regression_convergence.png`, `linear_regression_coefficients.png`, `linear_regression_holdout.png` |
| `classification.py` | `classification_metrics.json` | `classification_breast_cancer_confusion.png`, `classification_breast_cancer_comparison.png`, `classification_toy_text.png` |
| `clustering.py` | `clustering_metrics.json` | `clustering_blobs.png`, `clustering_image_quantization.png` |

Each JSON payload records `example`, `author`, `generated_by`, `seed` and
`environment` (Python, numpy, scikit-learn and matplotlib versions) alongside
the measurements for that script: generator parameters, per-estimator metrics,
cross-validation summaries, and quantization/compression numbers.

## How the measurements are produced

- Every number cited in the repository documentation for these examples was
  checked against the JSON files above when the prose was written. The prose is
  reviewed static text, not generated from the JSON at build time, so the cited
  numbers must be re-checked if the outputs are regenerated.
- All randomness (data generation, splits, folds, k-means restarts) comes from
  the recorded seed; no other seeds are used anywhere.
- Learned transforms are fitted on training data only — for the toy text study,
  the indicator token lists come from the training-split fit and each
  cross-validation fold refits within its training folds, so no model is fitted
  on the full corpus — and the same protocol is stated inside each JSON payload.
- Re-running with the same seed and library versions reproduces the same JSON
  and the same figures.
- Each script pins matplotlib's Agg backend, so figures are written as PNG
  files without needing a display, and each script also prints a compact
  measured summary to stdout.

## Limitations

- `linear_regression.py` studies one well-conditioned synthetic design with
  Gaussian noise. It compares solvers, not modeling strategies, and says
  nothing about real regression datasets.
- `classification.py` reports a single stratified split (plus cross-validation
  for model selection) on the bundled breast cancer data, so the numbers
  describe that split rather than the dataset as a whole. Gaussian naive Bayes
  is scale invariant; its scaler only keeps the pipeline convention.
- The toy message corpus is a handful of fictional sentences written for
  teaching. Scores on it demonstrate the mechanics of TF-IDF and multinomial
  naive Bayes on a tiny sample; they are not evidence of spam-filter quality and
  must not be compared with published SMS benchmarks.
- `clustering.py` reports what k-means does on separated blobs and on a
  synthetic raster. Image error is specific to this generated scene, its noise
  level and the chosen palette size, and the compression ratios are arithmetic
  for an indexed palette, not measured file sizes.
- k-means and all other estimators can behave differently under different
  library versions; the recorded `environment` block in each JSON file is part
  of the result.

## Data and provenance

- Breast cancer data: `sklearn.datasets.load_breast_cancer`, a copy bundled
  with scikit-learn; no download. The underlying dataset is the UCI Machine
  Learning Repository's **Breast Cancer Wisconsin (Diagnostic)** data set,
  originally created by Dr. William H. Wolberg, W. Nick Street, and Olvi L.
  Mangasarian at the University of Wisconsin; UCI lists it under CC BY 4.0
  (<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>).
  It is not redistributed here and is not covered by this repository's MIT
  license.
- Toy messages: original fictional text written for this repository.
- Blobs, linear data and the synthetic scene: generated by the scripts from the
  recorded seed.
- No course handouts, instructor notebooks, third-party figures, real message
  text, or photographs are included.
