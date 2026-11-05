"""Loading the saved pipeline, validating user input and making predictions.

This module has no Streamlit / FastAPI code, so the app, the REST API and the
unit tests all share exactly the same logic.
"""
import json
from pathlib import Path

from xgboost import XGBRegressor
import numpy as np
import pandas as pd

from .data import FEATURES, LABELS
from .features import ConcreteFeatures  # noqa: F401  (must be importable to unpickle the pipeline)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "concrete_pipeline.ubj"
META_PATH = ROOT / "models" / "metadata.json"

AGE_CURVE = [1, 3, 7, 14, 28, 56, 90, 180, 365]   # ages used for the strength-gain chart

# Example mixes for the demo buttons (every value is inside the training ranges).
PRESETS = {
    "Ordinary mix, 28 days": dict(cement=280.0, slag=0.0, fly_ash=0.0, water=190.0,
                                  superplasticizer=0.0, coarse_agg=970.0, fine_agg=800.0, age=28),
    "Fly-ash blended mix, 28 days": dict(cement=250.0, slag=0.0, fly_ash=100.0, water=175.0,
                                         superplasticizer=8.0, coarse_agg=1000.0, fine_agg=770.0, age=28),
    "Slag blended mix, 56 days": dict(cement=200.0, slag=150.0, fly_ash=0.0, water=180.0,
                                      superplasticizer=6.0, coarse_agg=950.0, fine_agg=800.0, age=56),
    "High-strength mix, 28 days": dict(cement=450.0, slag=0.0, fly_ash=0.0, water=160.0,
                                       superplasticizer=10.0, coarse_agg=950.0, fine_agg=780.0, age=28),
    "Ordinary mix, only 3 days old": dict(cement=280.0, slag=0.0, fly_ash=0.0, water=190.0,
                                          superplasticizer=0.0, coarse_agg=970.0, fine_agg=800.0, age=3),
}
DEFAULT_PRESET = "Ordinary mix, 28 days"


class InputValidationError(ValueError):
    """Raised when the inputs cannot be used; `.errors` holds readable messages."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))



class LoadedConcretePipeline:
    """Applies feature engineering and then predicts with XGBoost."""

    def __init__(self, feature_transformer, estimator):
        self.features = feature_transformer
        self.estimator = estimator

    def predict(self, X):
        X_transformed = self.features.transform(X)
        return self.estimator.predict(X_transformed)


def load_artifacts(model_path=MODEL_PATH, meta_path=META_PATH):
    """Load the portable XGBoost model and its metadata."""

    model_path = Path(model_path)
    meta_path = Path(meta_path)

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run notebooks/concrete_strength_pipeline.ipynb first."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if meta.get("model_format") != "xgboost_ubj":
        raise ValueError(
            "Unsupported model format. Regenerate the model using the updated notebook."
        )

    estimator = XGBRegressor()
    estimator.load_model(str(model_path))

    feature_transformer = ConcreteFeatures(
        add_engineered=meta["feature_engineering"]
    )

    model = LoadedConcretePipeline(feature_transformer, estimator)

    return model, meta



def validate_inputs(values, meta):
    """Check raw user inputs. Returns (clean_values, errors, warnings).

    errors   -> the prediction must not be made (missing / non-numeric / outside training range)
    warnings -> the prediction can be made but should be treated with extra care
    """
    errors, warnings, clean = [], [], {}
    ranges = meta["feature_ranges"]
    for name in FEATURES:
        label = LABELS[name]
        if values is None or name not in values or values[name] is None:
            errors.append(f"{label} is missing.")
            continue
        try:
            v = float(values[name])
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if not np.isfinite(v):
            errors.append(f"{label} must be a finite number.")
            continue
        lo, hi = ranges[name]["min"], ranges[name]["max"]
        if v < lo or v > hi:
            errors.append(f"{label} = {v:g} is outside the range the model was trained on ({lo:g} to {hi:g}).")
            continue
        clean[name] = v

    if not errors:
        binder = clean["cement"] + clean["slag"] + clean["fly_ash"]
        wb = clean["water"] / binder
        r = meta["water_binder_range"]
        if wb < r["min"] or wb > r["max"]:
            warnings.append(
                f"Water/binder ratio = {wb:.2f} is unusual (training data: {r['min']:.2f} to {r['max']:.2f}). "
                "The prediction may be unreliable for this combination of ingredients."
            )
    return clean, errors, warnings


def strength_category(mpa):
    """Very rough, indicative label (NOT a design-code concrete grade)."""
    if mpa < 20:
        return "Low (< 20 MPa)"
    if mpa < 40:
        return "Medium (20-40 MPa)"
    if mpa < 60:
        return "High (40-60 MPa)"
    return "Very high (60+ MPa)"


def predict_strength(model, meta, values):
    """Validate the raw inputs and predict compressive strength in MPa."""
    clean, errors, warnings = validate_inputs(values, meta)
    if errors:
        raise InputValidationError(errors)
    X = pd.DataFrame([clean], columns=FEATURES)
    pred = float(model.predict(X)[0])
    band = meta["residual_band_90"]
    binder = clean["cement"] + clean["slag"] + clean["fly_ash"]
    return {
        "strength_mpa": pred,
        "low_mpa": max(0.0, pred + band["low"]),     # 90 % of unseen mixes fall inside this band
        "high_mpa": pred + band["high"],
        "category": strength_category(pred),
        "water_cement": clean["water"] / clean["cement"],
        "water_binder": clean["water"] / binder,
        "warnings": warnings,
        "inputs": clean,
    }


def strength_curve(model, meta, values, ages=AGE_CURVE):
    """Predicted strength of the same mix at different curing ages."""
    clean, errors, _ = validate_inputs(values, meta)
    if errors:
        raise InputValidationError(errors)
    lo, hi = meta["feature_ranges"]["age"]["min"], meta["feature_ranges"]["age"]["max"]
    ages = [a for a in ages if lo <= a <= hi]
    X = pd.DataFrame([{**clean, "age": a} for a in ages], columns=FEATURES)
    return pd.DataFrame({"age": ages, "strength_mpa": model.predict(X)})
