import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import PRESETS, load_artifacts  # noqa: E402


@pytest.fixture(scope="session")
def artifacts():
    return load_artifacts()


@pytest.fixture()
def valid_mix():
    return dict(PRESETS["Ordinary mix, 28 days"])
