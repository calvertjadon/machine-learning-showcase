"""Shared fixtures for the tests that exercise the consumer-visible interface."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_showcase.features import chronological_split, prepare_plays
from nfl_showcase.models import fit_models
from nfl_showcase.smoke import fictional_plays


@pytest.fixture(scope="session")
def fictional_prepared() -> pd.DataFrame:
    """Prepared fictional plays shared by the feature and model tests."""
    return prepare_plays(fictional_plays())


@pytest.fixture(scope="session")
def fictional_split(fictional_prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The prepared plays split into train and test with the documented default cutoff."""
    return chronological_split(fictional_prepared)


@pytest.fixture(scope="session")
def trained_models(fictional_split: tuple[pd.DataFrame, pd.DataFrame]) -> dict[str, object]:
    """A small fitted model family shared across the model tests, with seed 42 and 8 trees."""
    train, _ = fictional_split
    return fit_models(train, seed=42, trees=8, jobs=1)
