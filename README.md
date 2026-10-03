# NFL play-type prediction — a leakage-conscious ML showcase

An original, reproducible machine-learning showcase that predicts the type of
play an NFL offense will call, using only information observable **before the
snap**.
The 2026 package reimplements the problem of a 2024 personal coursework
project as new code — developed with AI assistance under the author's
direction — and fixes the coursework's methodological weaknesses (row-level
leakage, previous-play values shifted across game boundaries, post-play inputs)
instead of reproducing them. It is not a copy of the coursework, and it is not
a production product.

## Objective

From pre-play context alone — down, distance, field position, clock, timeouts,
score, teams, and the immediately preceding play of the same game — classify
the play as one of six labels: `field_goal`, `pass`, `punt`, `qb_kneel`,
`qb_spike`, `run`. Evaluation is a **game-separated chronological holdout**:
models train on 2009–2016 games and are scored once on the 2017–2018 games, so
the numbers measure generalization to later seasons rather than memorization
of randomly split rows. All imputation and scaling are fitted on training rows
only, and current-play post-play outcomes are excluded. Previous-play yardage,
play type, and clock context are derived within the same game and are known
before the current snap.

## Measured results (real run)

Holdout: 66,965 plays from 480 games (2017-09-07 → 2018-12-17); training:
288,530 plays from 2,045 games (2009-09-10 → 2017-01-01), cutoff 2017-09-01.
Source values are in [`results/metrics.json`](results/metrics.json); values are
shown rounded to four decimals.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| `majority` (class-prior baseline) | 0.5255 | 0.1148 |
| `down_distance` (coaching-shorthand baseline) | 0.6069 | 0.3297 |
| `logistic_regression` | 0.6741 | **0.6734** |
| `random_forest` | **0.7093** | 0.6663 |

The baselines show why accuracy alone is misleading here: `pass` is 52.5% of
the holdout, so the majority baseline gets 0.5255 accuracy while scoring zero
F1 on five of six classes (macro-F1 0.1148). The random forest has the
**highest accuracy**, but logistic regression has the **higher macro-F1** —
neither dominates every class, and there is no universal best model in this
run. The clearest difference is the rarest class: the forest recovers only
**2 of the 121** holdout `qb_spike` plays (recall 0.0165, precision 1.0, F1
0.0325), while logistic regression recovers 34 of 121 (recall 0.2810, F1
0.3134). No calibration or threshold tuning was applied.

![Accuracy and macro-F1 by model](results/comparison.png)

![Per-class F1 by model](results/per_class_f1.png)

![Random forest holdout confusion matrix](results/random_forest_confusion.png)

Run context: seed 42, 200 trees, max depth 18, min leaf 5, 2 jobs. Full
protocol, per-class supports, and limitations — including uncalibrated
probabilities — are in [`docs/experiment.md`](docs/experiment.md).

## Quickstart and reproducible gates

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run nfl-showcase smoke
```

`pytest` runs 59 tests on original fictional fixtures; `smoke` runs a
deterministic mechanics check on generated fictional play data (no raw dataset
needed) and makes no performance claim.

The three self-contained examples are separate from the NFL package and run on
bundled or generated data:

```bash
uv run python examples/linear_regression.py --output-dir results/examples
uv run python examples/classification.py --output-dir results/examples
uv run python examples/clustering.py --output-dir results/examples
```

## Real data: acquisition, training, evaluation

The NFL data is **not** distributed with this repository. Obtain the public
*NFL Play by Play 2009-2018 (v5)* CSV from
[Kaggle](https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016)
and verify it against the documented size and SHA-256 — full instructions,
rights discussion, and the exact file facts are in
[`docs/data.md`](docs/data.md). The commands below assume the extracted CSV is
kept at `data/NFL Play by Play 2009-2018 (v5).csv` (`data/` is git-ignored);
adjust the path if you store it elsewhere.

```bash
# Fit and validate the four models; writes artifacts/training.json and a local bundle.
uv run nfl-showcase train \
  --data "data/NFL Play by Play 2009-2018 (v5).csv" \
  --exclude-ambiguous-games \
  --output-dir artifacts --holdout-start 2017-09-01 \
  --seed 42 --trees 200 --jobs 2

# Score the saved bundle on its own hash-matched holdout; writes results/metrics.json + figures.
uv run nfl-showcase evaluate \
  --data "data/NFL Play by Play 2009-2018 (v5).csv" \
  --model artifacts/models.joblib \
  --output-dir results
```

`--exclude-ambiguous-games` is required for this release: after collapsing
2,393 exact-duplicate rows, 22 conflicting play identifiers remain in one
game, and the explicit cleaning step drops that whole game (268 rows) rather
than guessing; 446,710 of 449,371 raw rows are retained. Without the flag,
`train` rejects the source with an actionable error. `evaluate` refuses a CSV
whose SHA-256 does not match the hash recorded at training time.

## Inference with honest pre-play context

`predict` scores explicit hypothetical contexts with the saved random forest
and prints labels with probabilities:

```bash
uv run nfl-showcase predict --model artifacts/models.joblib --input examples/pre_play_context.json
```

[`examples/pre_play_context.json`](examples/pre_play_context.json) is an
original, internally consistent **hypothetical** context — Miami trailing by 3
with the ball, 2nd-and-4 at the Buffalo 38, 5:00 left in the fourth quarter,
after a 6-yard run that kept the clock running. It contains all 16 required
feature columns and is not a row from the dataset. There is deliberately **no
mean-fill fallback for missing game context**: the CLI never substitutes a
value of its own, so all 16 values must be supplied (JSON `null` only forwards
a value to the training-fitted imputer, which is not a substitute for honest
context). There is also no hosted or interactive prediction widget and no
obsolete demo code in this repository; inference is batch CLI only.

## Repository layout

| Path | Contents |
| --- | --- |
| [`src/nfl_showcase/`](src/nfl_showcase/) | Feature engineering, models, and the `train`/`evaluate`/`predict`/`smoke` CLI. |
| [`docs/index.html`](docs/index.html) | Static report page over the measured results and figures. |
| [`docs/data.md`](docs/data.md) | Dataset source, verification, acquisition, and cleaning evidence. |
| [`docs/features.md`](docs/features.md) | Full pre-play feature audit and rejection list. |
| [`docs/experiment.md`](docs/experiment.md) | Protocol, split, metrics, historical comparison, limitations. |
| [`docs/provenance.md`](docs/provenance.md) | Authorship, rights, third-party licenses, data rights detail. |
| [`examples/`](examples/) | Three documented examples plus the sample inference context; see [`examples/README.md`](examples/README.md). |
| [`tests/`](tests/) | Original test suite on fictional fixtures. |
| [`results/`](results/) | Aggregate metrics JSON and generated figures (published); [`results/examples/`](results/examples/) holds example outputs. |
| [`LICENSE`](LICENSE) | MIT, scoped to this repository's original work. |

### Static report

The report is a read-only static page over the published aggregate results —
not a prediction service. Serve it from the repository root so its relative
`results/` fetches resolve, then open the `docs/` path:

```bash
uv run python -m http.server 8000 --bind 127.0.0.1
# open http://127.0.0.1:8000/docs/
```

## Provenance, rights, and license

- **Original work only.** The MIT [`LICENSE`](LICENSE) covers the code, tests,
  documentation, examples, and aggregate results created for this repository.
  No instructor material, 2024 notebooks, reports, photos, or old model files
  are included.
- **AI-assisted refresh.** The 2026 implementation and documentation were
  developed with AI assistance under the author's direction; this project does
  not claim that every new line or interface was personally hand-authored. The
  underlying 2024 project is the author's own coursework; the verified 2026
  results are new.
- **No data redistribution.** The raw NFL dataset, model bundles, and
  row-level extracts are never published. Kaggle's metadata lists the dataset
  license as `Unknown`; upstream rights are unresolved and this project claims
  none — see [`docs/provenance.md`](docs/provenance.md).
- **Historical numbers are narrative only.** Any 2024 figures (for example the
  reported micro-F1 around 0.75–0.77) come from a different, unsafe protocol
  and are **not comparable** with the results above; see
  [`docs/experiment.md`](docs/experiment.md#historical-2024-comparison-explicitly-not-comparable).

NFL and team names are used nominatively to describe the public dataset. No
NFL affiliation or endorsement is implied.
