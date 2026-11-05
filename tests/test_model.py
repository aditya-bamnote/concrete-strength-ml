"""The saved model should behave like real concrete and keep its quality."""
import pandas as pd
from sklearn.metrics import r2_score

from src.data import FEATURES, TARGET, load_clean
from src.predict import PRESETS, predict_strength, strength_curve


def test_artifacts_load_and_metadata_is_complete(artifacts):
    model, meta = artifacts
    for key in ("model_name", "test_metrics", "residual_band_90", "feature_ranges", "water_binder_range"):
        assert key in meta
    assert set(meta["feature_ranges"]) == set(FEATURES)


def test_reported_test_quality_guard(artifacts):
    _, meta = artifacts
    assert meta["test_metrics"]["R2"] > 0.85          # fails loudly if a re-training made the model much worse
    assert meta["test_metrics"]["MAE"] < 5.0


def test_pipeline_accepts_raw_columns_and_fits_the_data_well(artifacts):
    model, _ = artifacts
    df = load_clean()
    pred = model.predict(df[FEATURES])
    assert len(pred) == len(df)
    assert r2_score(df[TARGET], pred) > 0.9           # training-set sanity check (it was refit on all rows)


def test_prediction_is_sensible_and_includes_a_range(artifacts, valid_mix):
    model, meta = artifacts
    res = predict_strength(model, meta, valid_mix)
    assert 5 < res["strength_mpa"] < 90
    assert res["low_mpa"] <= res["strength_mpa"] <= res["high_mpa"]
    assert res["low_mpa"] >= 0
    assert res["category"]


def test_strength_grows_with_curing_age(artifacts, valid_mix):
    model, meta = artifacts
    curve = strength_curve(model, meta, valid_mix, ages=[3, 7, 28, 90])
    assert list(curve["age"]) == [3, 7, 28, 90]
    assert curve["strength_mpa"].is_monotonic_increasing


def test_more_water_means_weaker_concrete(artifacts, valid_mix):
    model, meta = artifacts
    low = predict_strength(model, meta, {**valid_mix, "water": 160.0})["strength_mpa"]
    high = predict_strength(model, meta, {**valid_mix, "water": 220.0})["strength_mpa"]
    assert low > high


def test_high_strength_preset_beats_ordinary_preset(artifacts):
    model, meta = artifacts
    ordinary = predict_strength(model, meta, PRESETS["Ordinary mix, 28 days"])["strength_mpa"]
    strong = predict_strength(model, meta, PRESETS["High-strength mix, 28 days"])["strength_mpa"]
    assert strong > ordinary


def test_prediction_does_not_change_input_dict(artifacts, valid_mix):
    model, meta = artifacts
    before = dict(valid_mix)
    predict_strength(model, meta, valid_mix)
    assert valid_mix == before
