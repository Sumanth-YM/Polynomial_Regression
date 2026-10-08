"""Shared helpers for the polynomial regression assignment: data loading, models, CV setup."""
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
RIDGE_ALPHAS = np.logspace(-8, 3, 23)

FEATURES = {
    "var1": ["x1", "x2", "x3", "x4", "x5", "x6"],
    "var2": ["x1", "x2", "x3"],
}
# Degree lists used by the final script. Lasso is slow at high degree, so its list is shorter.
DEFAULT_RIDGE_DEGREES = {"var1": "1-10", "var2": "4-20"}
DEFAULT_LASSO_DEGREES = {"var1": "3-8", "var2": "6,8,10,12,14,16,18,20"}
# Degree lists used by the experiment scripts.
DEFAULT_EXPERIMENT_DEGREES = {"var1": "3-8", "var2": "6-14"}


def parse_degrees(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def load_data(roll, var, data_dir="."):
    """Returns X_train, y_train, X_test, feature names for one problem."""
    d = Path(data_dir)
    train_path = d / f"{roll}_train_{var}.csv"
    test_path = d / f"{roll}_test_{var}.csv"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Check --roll and --data-dir.")
    feats = FEATURES[var]
    tr, te = pd.read_csv(train_path), pd.read_csv(test_path)
    return tr[feats].values, tr["y"].values, te[feats].values, feats


def get_kfold():
    """Identical folds everywhere, so all models are compared on the same splits."""
    return KFold(N_SPLITS, shuffle=True, random_state=SEED)


def make_ridge(degree, x_range=1.0, restandardize=False, inner_cv=None):
    """MinMax scale to [-x_range, x_range] -> polynomial terms -> (optional StandardScaler) -> RidgeCV."""
    steps = [
        MinMaxScaler(feature_range=(-x_range, x_range)),
        PolynomialFeatures(degree=degree, include_bias=False),
    ]
    if restandardize:
        steps.append(StandardScaler())
    if inner_cv is None:
        steps.append(RidgeCV(alphas=RIDGE_ALPHAS, gcv_mode="svd"))
    else:
        steps.append(RidgeCV(alphas=RIDGE_ALPHAS, cv=inner_cv))
    return make_pipeline(*steps)


def make_lasso(n_jobs=-1):
    """LassoCV with an inner 5-fold CV for alpha."""
    if "n_alphas" in inspect.signature(LassoCV).parameters:
        grid = {"n_alphas": 50}
    else: 
        grid = {"alphas": 50}
    return LassoCV(cv=5, n_jobs=n_jobs, max_iter=50000, eps=1e-3, **grid)


def make_lasso_pipeline(degree, n_jobs=-1, x_range=1.0):
    return make_pipeline(
        MinMaxScaler(feature_range=(-x_range, x_range)),
        PolynomialFeatures(degree=degree, include_bias=False),
        make_lasso(n_jobs=n_jobs),
    )


def poly_transformer(X_train, degree, x_range=1.0):
    """Fitted scaler + polynomial expansion (fit on the training rows only)."""
    t = make_pipeline(
        MinMaxScaler(feature_range=(-x_range, x_range)),
        PolynomialFeatures(degree=degree, include_bias=False),
    )
    return t.fit(X_train)


def se(values):
    values = np.asarray(values, dtype=float)
    return values.std() / np.sqrt(len(values))
