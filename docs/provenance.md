# Provenance and rights

This repository is an original, independent machine-learning showcase
published by Jadon Calvert in 2026. It reimplements, as new code, the problem
of a machine-learning course project he completed in 2024. The 2026
implementation and its documentation were created for this repository with
extensive AI assistance under his direction and review; the project does not
claim that every new line or interface was personally hand-authored, and it is
not a copy, excerpt, or republication of the 2024 coursework. The 2024 project
is his own coursework; the results verified here are new.

The private 2024 coursework archive is not published, is not modified by this
project, and shares no history with this repository.

## Original work

Original to this repository (covered by the MIT license in `../LICENSE`):

- `src/nfl_showcase/` — the feature-preparation, modeling, and CLI package,
  including its leakage-conscious temporal evaluation protocol.
- `tests/` — the test suite.
- `.github/workflows/ci.yml`.
- `examples/` — the three self-contained demo scripts and the sample
  inference context.
- `docs/` — this documentation, including `index.html`.
- `README.md` — the project landing page.
- Aggregate outputs under `results/` (metrics, figures) produced by the
  commands in the README.

The 2024 experiment used a different, unsafe setup (a notebook workflow with a
random row-level split and transforms fitted outside the training fold).
Figures quoted from it — for example the reported micro-F1 of about 0.75-0.77
mentioned as history in `experiment.md` — are narrative history, not results
of this repository, are not regenerated here, and are not comparable
head-to-head with the current measurements.

The MIT license and the authorship statements above cover this repository's
original work as a whole; they do not assert that every individual line was
typed without tooling assistance.

## Explicitly excluded material

The following are deliberately not included, copied, adapted, or
redistributed here:

- Instructor-provided and course-provided materials: lecture notes, slides,
  lab handouts, assignment prompts and summaries, rubrics, guidelines, and
  any instructor-authored code or notebook text. No authorship of any course
  material is claimed.
- The 2024 coursework notebooks and their saved outputs.
- The 2024 final-project report, proposal, and reading-report documents,
  including any third-party article text or paraphrase they contain.
- Third-party papers, articles, and PDFs/EPUBs distributed with the course.
- Third-party images and photographs used in course exercises.
- Course-associated datasets that this showcase excludes and does not
  redistribute (for example, the UCI SMS Spam Collection,
  https://archive.ics.uci.edu/dataset/228/sms+spam+collection); the examples
  use original synthetic or library-bundled data instead. The exclusion is a
  scope decision for this repository, not a statement that these datasets are
  legally nonredistributable; consult the linked source page for the terms
  that apply.
- The raw NFL CSV, all derived CSVs/dataframes, and every model artifact from
  2024 (trained `.pkl` files, random-search checkpoints, and similar). Model
  files can execute code when loaded; this repository neither ships nor
  accepts them.
- Personal or identifying information.

Where documentation mentions historical coursework at all, it does so only as
narrative context for why the modern implementation exists.

None of the excluded material is required to build, test, run, or understand
this repository. The NFL pipeline needs only the public dataset described in
`data.md`, and the examples generate or load their own data.

## Relationship to course staff and institutions

This project is not affiliated with, endorsed by, or connected to any
university, course, or instructor, and it makes no attribution of any design
to course staff. The interfaces, evaluation protocol, and implementation
documented here are the author's own decisions for this repository.

## Third-party software

This repository's MIT license covers only its original work. Third-party
components keep their own licenses:

| Component | Used for | License | Primary source |
| --- | --- | --- | --- |
| Python 3.12 (project requires >=3.12,<3.13) | runtime | Python Software Foundation License Version 2 | https://docs.python.org/3/license.html |
| scikit-learn 1.5.2 | models, pipelines, metrics | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| NumPy 1.26.4 | numeric arrays | BSD-3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| pandas 2.2.3 | data frames | BSD-3-Clause | https://github.com/pandas-dev/pandas/blob/main/LICENSE |
| matplotlib 3.9.2 | figures | Matplotlib license agreement | https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE |
| joblib 1.6.0 | model-bundle persistence | BSD-3-Clause | https://github.com/joblib/joblib/blob/main/LICENSE.txt |
| pytest 9.1.1 (dev) | test runner | MIT | https://github.com/pytest-dev/pytest/blob/main/LICENSE |
| ruff 0.15.22 (dev) | lint and format | MIT | https://github.com/astral-sh/ruff/blob/main/LICENSE |
| hatchling (build backend) | packaging | MIT | https://github.com/pypa/hatch/blob/master/backend/LICENSE.txt |
| uv (workflow tool) | environment and lock management | Apache-2.0 OR MIT | https://github.com/astral-sh/uv/blob/main/LICENSE-MIT and https://github.com/astral-sh/uv/blob/main/LICENSE-APACHE |

Versions are pinned in `pyproject.toml`; `uv.lock` holds the full resolution.
These packages are not redistributed in this repository; they are fetched from
package indexes at install time. Nothing here re-licenses any dependency.

## Datasets

- **NFL play-by-play (2009-2018)** — not distributed with this repository.
  Kaggle's metadata lists the dataset license as `Unknown` (no license
  declared), the upstream data repositories declare no data license, and the
  data was originally scraped from NFL.com. This project claims no permission
  or rights in it. See `data.md` for source, download, and verification
  details.
- **scikit-learn bundled breast cancer dataset** — used read-only by
  `examples/classification.py`. It is a copy of the UCI ML Breast Cancer
  Wisconsin (Diagnostic) dataset
  (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
  which UCI states is licensed CC BY 4.0. It is not redistributed here and is
  not covered by this repository's MIT license; consult the UCI page and
  scikit-learn's own terms.
- **Data generated by this repository** — the synthetic regression data,
  blobs, and synthetic raster in `examples/`, the fictional
  spam/ham message corpus in `examples/classification.py`, and the fictional
  play data used by the smoke command and tests are original to this
  repository and covered by its MIT license as part of its original code and
  analysis.

## Data rights detail

The primary-source record for the NFL dataset is unusually thin, and this
document does not try to resolve it:

- Kaggle's dataset metadata reports the license as `Unknown`:
  https://www.kaggle.com/api/v1/datasets/view/maxhorowitz/nflplaybyplay2009to2016
- The dataset was generated with **nflscrapR**, which "uses an API maintained
  by the NFL to scrape, clean, parse, and output clean datasets" (dataset
  description, same record). The nflscrapR R package declares `License: CC0`
  in its `DESCRIPTION`
  (https://github.com/maksimhorowitz/nflscrapR/blob/master/DESCRIPTION), but
  that covers the package code, not the scraped data. The companion data
  repository https://github.com/ryurko/nflscrapR-data contains no license
  file, and GitHub's metadata reports no license for it.
- The expected-points and win-probability columns in the file come from the
  nflscrapR research project: Yurko, Ventura & Horowitz, *nflWAR: A
  Reproducible Method for Offensive Player Evaluation in Football*,
  https://arxiv.org/abs/1802.00998.
- NFL.com's Terms and Conditions (updated 2024-05-16,
  https://www.nfl.com/legal/terms/) state that the NFL owns all right, title
  and interest in content and data "included or displayed in or through,
  provided with or through the use of, or generated from" its services, and
  prohibit systematic retrieval of data without prior written consent.
- Kaggle's own Terms of Use govern the act of downloading:
  https://www.kaggle.com/terms.

Because the upstream rights position is unresolved, this repository publishes
only original code and aggregate derived results. It does not redistribute
the raw data, rows, play descriptions, or player-identifying fields, and it
makes no claim that any particular reuse of the dataset is permitted.

Whether the dataset's publisher had the right to redistribute scraped NFL
data is unknown; that uncertainty is acknowledged rather than papered over.
It does not restrict the MIT licensing of the original code and analysis
here, which are independent of the dataset.

## Claims and metrics

- Metrics published under `results/` are produced by the code in this
  repository from the documented data and the documented game-separated
  chronological holdout. No result is hand-entered or copied from the 2024
  experiment.
- There is no test-set model selection; reported numbers are single-holdout
  measurements, not benchmark claims about football prediction in general.
- The smoke fixture and examples make no performance claims about real NFL
  data; the smoke report explicitly measures mechanics only.
- NFL and team names are used nominatively to describe the dataset's
  contents. No endorsement by or affiliation with the NFL is implied, and no
  NFL marks are licensed by this project.
- This repository contains no rows from the raw dataset and no
  player-identifying information in its documentation, reports, or figures.

## Primary sources

- Kaggle dataset page: https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016
- Kaggle dataset metadata (license `Unknown`, version history): https://www.kaggle.com/api/v1/datasets/view/maxhorowitz/nflplaybyplay2009to2016
- nflscrapR scraper (package `License: CC0`): https://github.com/maksimhorowitz/nflscrapR
- nflscrapR data repository (no license declared): https://github.com/ryurko/nflscrapR-data
- nflWAR paper (expected points / win probability models): https://arxiv.org/abs/1802.00998
- NFL.com Terms and Conditions: https://www.nfl.com/legal/terms/
- Kaggle Terms of Use: https://www.kaggle.com/terms
- Kaggle CLI documentation (authentication, downloads): https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md
- UCI Breast Cancer Wisconsin (Diagnostic), CC BY 4.0: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- UCI SMS Spam Collection (excluded): https://archive.ics.uci.edu/dataset/228/sms+spam+collection
