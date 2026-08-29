"""Shared fixtures: bundled data paths and one full-pipeline run per session.

The golden values were produced with the pinned full config, so the golden
tests run the identical full pipeline once (session-scoped) and every
HMM-level test reuses those fits instead of refitting.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def index_returns() -> np.ndarray:
    return pd.read_csv(DATA_DIR / "market_index.csv")["ret"].to_numpy()


@pytest.fixture(scope="session")
def golden() -> dict:
    with open(DATA_DIR / "golden" / "golden.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def pipeline_full() -> dict:
    from regime import run_full_pipeline

    return run_full_pipeline(DATA_DIR)
