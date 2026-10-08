"""Experiment 1: baselines and scaling choices (supports the 'why ridge' and 'why this scaling' sections).

Compares, over several degrees, with identical 5-fold CV splits:
  ols                         plain least squares (no regularization)
  ridge_restandardize_gcv     ridge, StandardScaler applied after the polynomial expansion
  ridge_plain_gcv             ridge, no re-standardization (inputs in [-1, 1])
  ridge_restandardize_innercv same as the first ridge, alpha chosen by inner 5-fold CV
  ridge_plain_gcv_trimmed     ridge_plain_gcv trained on rows with |z| < 3 only (outlier removal),
                              evaluated on ALL validation rows

Reports MSE on all rows, MSE on rows with |z(y)| < 3, the median |residual|, and median/RMSE
(about 0.67 when the noise is Gaussian, much smaller when a few large residuals dominate).

    python exp1_baselines_and_scaling.py --roll BT2024072 --var var1
    python exp1_baselines_and_scaling.py --roll BT2024072 --var var2 --degrees 4-14
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

import polyutils as pu

VARIANTS = [
    "ols",
    "ridge_restandardize_gcv",
    "ridge_plain_gcv",
    "ridge_restandardize_innercv",
    "ridge_plain_gcv_trimmed",
]


def build(name, degree):
    if name == "ols":
        return make_pipeline(MinMaxScaler((-1, 1)), PolynomialFeatures(degree, include_bias=False),
                             LinearRegression())
    if name == "ridge_restandardize_gcv":
        return pu.make_ridge(degree, restandardize=True)
    if name in ("ridge_plain_gcv", "ridge_plain_gcv_trimmed"):
        return pu.make_ridge(degree)
    if name == "ridge_restandardize_innercv":
        return pu.make_ridge(degree, restandardize=True, inner_cv=5)
    raise ValueError(name)


def fit_fold(name, degree, tr, va, X, y):
    if name.endswith("_trimmed"):
        data = np.c_[X[tr], y[tr]]
        z = np.abs((data - data.mean(0)) / data.std(0))
        keep = tr[(z < 3).all(axis=1)]
    else:
        keep = tr
    model = build(name, degree).fit(X[keep], y[keep])
    return name, degree, va, model.predict(X[va])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", required=True)
    ap.add_argument("--var", choices=["var1", "var2"], required=True)
    ap.add_argument("--degrees", default=None, help="e.g. 3-8 or 4,6,8")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    degrees = pu.parse_degrees(args.degrees or pu.DEFAULT_EXPERIMENT_DEGREES[args.var])
    X, y, _, _ = pu.load_data(args.roll, args.var, args.data_dir)
    splits = list(pu.get_kfold().split(X))
    print(f"{args.var}: {len(y)} rows, degrees {degrees}")

    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(fit_fold)(name, d, tr, va, X, y)
        for name in VARIANTS for d in degrees for tr, va in splits
    )
    oof = {}
    for name, d, va, pred in out:
        oof.setdefault((name, d), np.zeros(len(y)))[va] = pred

    keep_y = np.abs((y - y.mean()) / y.std()) < 3
    rows = []
    for (name, d), p in sorted(oof.items(), key=lambda kv: (VARIANTS.index(kv[0][0]), kv[0][1])):
        r = y - p
        rmse = np.sqrt(np.mean(r ** 2))
        rows.append({"variant": name, "degree": d, "mse_all": np.mean(r ** 2),
                     "mse_z3": np.mean(r[keep_y] ** 2), "median_abs_res": np.median(np.abs(r)),
                     "median_over_rmse": np.median(np.abs(r)) / rmse})
    res = pd.DataFrame(rows)
    print("\n" + res.round(4).to_string(index=False))
    res.to_csv(f"results_exp1_{args.var}.csv", index=False)
    print(f"\nsaved results_exp1_{args.var}.csv")


if __name__ == "__main__":
    main()
