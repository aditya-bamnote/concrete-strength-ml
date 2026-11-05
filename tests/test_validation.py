"""Input validation: the app must reject bad input with a clear message, never crash."""
import math

import pytest

from src.predict import PRESETS, InputValidationError, predict_strength, validate_inputs


def test_valid_input_passes(artifacts, valid_mix):
    _, meta = artifacts
    clean, errors, warnings = validate_inputs(valid_mix, meta)
    assert errors == [] and warnings == []
    assert clean["age"] == 28


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_is_inside_the_training_ranges(artifacts, name):
    _, meta = artifacts
    _, errors, _ = validate_inputs(PRESETS[name], meta)
    assert errors == []


def test_missing_field(artifacts, valid_mix):
    _, meta = artifacts
    del valid_mix["water"]
    _, errors, _ = validate_inputs(valid_mix, meta)
    assert any("missing" in e for e in errors)


def test_none_value(artifacts, valid_mix):
    _, meta = artifacts
    valid_mix["cement"] = None
    _, errors, _ = validate_inputs(valid_mix, meta)
    assert any("missing" in e for e in errors)


def test_non_numeric_value(artifacts, valid_mix):
    _, meta = artifacts
    valid_mix["cement"] = "lots"
    _, errors, _ = validate_inputs(valid_mix, meta)
    assert any("must be a number" in e for e in errors)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nan_and_infinity(artifacts, valid_mix, bad):
    _, meta = artifacts
    valid_mix["water"] = bad
    _, errors, _ = validate_inputs(valid_mix, meta)
    assert errors


@pytest.mark.parametrize("name, value", [("cement", -10), ("cement", 5000), ("age", 0), ("age", 4000), ("water", 20)])
def test_out_of_range_values_are_rejected(artifacts, valid_mix, name, value):
    _, meta = artifacts
    valid_mix[name] = value
    _, errors, _ = validate_inputs(valid_mix, meta)
    assert any("outside the range" in e for e in errors)


def test_predict_raises_on_invalid_input(artifacts, valid_mix):
    model, meta = artifacts
    valid_mix["age"] = -5
    with pytest.raises(InputValidationError):
        predict_strength(model, meta, valid_mix)


def test_unusual_water_binder_ratio_gives_a_warning_not_an_error(artifacts, valid_mix):
    _, meta = artifacts
    # each value is individually inside its range, but 247 kg water for ~102 kg binder is far outside the data
    valid_mix.update(cement=meta["feature_ranges"]["cement"]["min"], slag=0.0, fly_ash=0.0,
                     water=meta["feature_ranges"]["water"]["max"])
    _, errors, warnings = validate_inputs(valid_mix, meta)
    assert errors == []
    assert warnings and "Water/binder" in warnings[0]
