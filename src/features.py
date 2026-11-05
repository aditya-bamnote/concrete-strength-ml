"""Feature engineering, written as a scikit-learn transformer so that it is
saved *inside* the final pipeline and the app can accept raw user inputs."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_FEATURES = [
    "cement", "slag", "fly_ash", "water",
    "superplasticizer", "coarse_agg", "fine_agg", "age",
]
ENGINEERED = ["binder", "water_cement", "water_binder", "log_age"]


class ConcreteFeatures(BaseEstimator, TransformerMixin):
    """Adds domain features to the 8 raw inputs.

    binder        = cement + slag + fly ash (total cementitious material)
    water_cement  = water / cement            (classic strength driver)
    water_binder  = water / binder
    log_age       = log(1 + age)              (strength grows ~ log of age)

    Set add_engineered=False to get only the raw columns (used for the
    'with vs without feature engineering' comparison in the notebook).
    """

    def __init__(self, add_engineered=True):
        self.add_engineered = add_engineered

    def fit(self, X, y=None):
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X[RAW_FEATURES]
        if not self.add_engineered:
            return X
        binder = X["cement"] + X["slag"] + X["fly_ash"]
        X["binder"] = binder
        X["water_cement"] = X["water"] / X["cement"].clip(lower=1e-6)
        X["water_binder"] = X["water"] / binder.clip(lower=1e-6)
        X["log_age"] = np.log1p(X["age"])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_FEATURES + (ENGINEERED if self.add_engineered else []))
