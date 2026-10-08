"""Experiment 2: which regularizer fits this dataset? (supports the model-choice section)

Compares, with identical 5-fold CV splits:
  ridge_range_1.0 / 0.7 / 0.5   ridge with inputs scaled to [-r, r] (a smaller r penalizes
                                high-degree terms more strongly)
  lasso                          LassoCV (inner 5-fold), sparse polynomial

Also prints how many terms Lasso keeps. Dense signals favour ridge, sparse ones favour Lasso.

    python exp2_regularization_comparison.py --roll BT2024072 --var var1
    python exp2_regularization_comparison.py --roll BT2024072 --var var2 --degrees 8,10,12,14
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

import polyutils as pu

RIDGE_RANGES = [1.0, 0.7, 0.5]


def ridge_task(degree, x_range, tr, va, X, y):
    m = pu.make_ridge(degree, x_range=x_range).fit(X[tr], y[tr])
    return ("ridge_range_" + str(x_range), degree, mean_squared_error(y[va], m.predict(X[va])), np.nan)


def lasso_task(degree, tr, va, X, y):
    t = pu.poly_transformer(X[tr], degree)
    Ztr, Zva = t.transform(X[tr]), t.transform(X[va])
    lasso = pu.make_lasso(n_jobs=1).fit(Ztr, y[tr])
    return ("lasso", degree, mean_squared_error(y[va], lasso.predict(Zva)), np.count_nonzero(lasso.coef_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", required=True)
    ap.add_argument("--var", choices=["var1", "var2"], required=True)
    ap.add_argument("--degrees", default=None, help="e.g. 4-7 or 8,10,12")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    degrees = pu.parse_degrees(args.degrees or pu.DEFAULT_EXPERIMENT_DEGREES[args.var])
    X, y, _, _ = pu.load_data(args.roll, args.var, args.data_dir)
    splits = list(pu.get_kfold().split(X))
    print(f"{args.var}: {len(y)} rows, degrees {degrees}")

    tasks = [delayed(lasso_task)(d, tr, va, X, y) for d in reversed(degrees) for tr, va in splits]
    tasks += [delayed(ridge_task)(d, r, tr, va, X, y)
              for r in RIDGE_RANGES for d in degrees for tr, va in splits]
    out = Parallel(n_jobs=args.jobs, verbose=5)(tasks)

    df = pd.DataFrame(out, columns=["model", "degree", "fold_mse", "nnz"])
    res = (df.groupby(["model", "degree"])
             .agg(mse=("fold_mse", "mean"), se=("fold_mse", pu.se), terms_kept=("nnz", "mean"))
             .reset_index())
    order = {"ridge_range_1.0": 0, "ridge_range_0.7": 1, "ridge_range_0.5": 2, "lasso": 3}
    res = res.sort_values(by=["model", "degree"], key=lambda c: c.map(order) if c.name == "model" else c)
    print("\n" + res.round(4).to_string(index=False))
    res.to_csv(f"results_exp2_{args.var}.csv", index=False)
    best = res.loc[res["mse"].idxmin()]
    print(f"\nbest: {best['model']} at degree {int(best['degree'])}, CV MSE {best['mse']:.4f}")
    print(f"saved results_exp2_{args.var}.csv")


if __name__ == "__main__":
    main()
