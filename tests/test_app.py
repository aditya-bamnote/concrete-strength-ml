"""Smoke test of the Streamlit UI using Streamlit's built-in AppTest runner."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_starts_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Concrete Compressive Strength Predictor" in at.title[0].value


def test_predict_button_shows_a_prediction():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button[0].click().run()                    # the form's "Predict strength" button
    assert not at.exception
    assert not at.error
    labels = [m.label for m in at.metric]
    assert "Predicted compressive strength" in labels
    value = float(at.metric[labels.index("Predicted compressive strength")].value.split()[0])
    assert 5 < value < 90


def test_choosing_a_preset_changes_the_inputs():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox[0].select("High-strength mix, 28 days").run()
    assert at.session_state["in_cement"] == 450.0
    assert not at.exception
