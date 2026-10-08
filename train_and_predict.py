"""Final training + inference script.

For each problem (var1, var2) it
  1. cross-validates ridge and Lasso over a range of polynomial degrees
     (5-fold CV, identical folds for every model), tracking both Train and Val MSE,
  2. picks the best model by CV Val MSE, then the simplest degree within one standard error of it,
  3. reports honest out-of-fold Val MSE / R2 and total Train MSE / R2 for that choice,
  4. plots a Bias-Variance tradeoff curve (Train vs Val MSE),
  5. refits on ALL training rows and writes the test predictions.

    python train_and_predict.py --roll BT2024072
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score

import polyutils as pu


def ridge_fold(X, y, tr, va, d):
    m = pu.make_ridge(d).fit(X[tr], y[tr])
    train_mse = mean_squared_error(y[tr], m.predict(X[tr]))
    val_mse = mean_squared_error(y[va], m.predict(X[va]))
    return ("ridge", d, train_mse, val_mse, np.nan)


def lasso_fold(X, y, tr, va, d):
    t = pu.poly_transformer(X[tr], d)
    Ztr, Zva = t.transform(X[tr]), t.transform(X[va])
    lasso = pu.make_lasso(n_jobs=1).fit(Ztr, y[tr])  # parallelism is across (degree, fold)
    train_mse = mean_squared_error(y[tr], lasso.predict(Ztr))
    val_mse = mean_squared_error(y[va], lasso.predict(Zva))
    return ("lasso", d, train_mse, val_mse, np.count_nonzero(lasso.coef_))


def cv_table(X, y, ridge_degrees, lasso_degrees, jobs):
    splits = list(pu.get_kfold().split(X))
    tasks = [delayed(lasso_fold)(X, y, tr, va, d) for d in reversed(lasso_degrees) for tr, va in splits]
    tasks += [delayed(ridge_fold)(X, y, tr, va, d) for d in ridge_degrees for tr, va in splits]
    out = Parallel(n_jobs=jobs, verbose=5)(tasks)
    df = pd.DataFrame(out, columns=["model", "degree", "fold_train_mse", "fold_val_mse", "nnz"])
    return (df.groupby(["model", "degree"])
              .agg(train_mse=("fold_train_mse", "mean"), 
                   val_mse=("fold_val_mse", "mean"), 
                   val_se=("fold_val_mse", pu.se), 
                   terms_kept=("nnz", "mean"))
              .reset_index())


def fit_predict(kind, degree, X, y, X_new, feats):
    """Fit on (X, y), predict X_new. Returns (test_preds, train_preds, info_dict)."""
    if kind == "ridge":
        m = pu.make_ridge(degree).fit(X, y)
        return m.predict(X_new), m.predict(X), {"alpha": float(m[-1].alpha_)}

    t = pu.poly_transformer(X, degree)
    Z, Zn = t.transform(X), t.transform(X_new)
    lasso = pu.make_lasso(n_jobs=-1).fit(Z, y)
    names = t[-1].get_feature_names_out(feats)
    info = {
        "alpha": float(lasso.alpha_),
        "alpha_at_grid_edge": bool(np.isclose(lasso.alpha_, lasso.alphas_[-1])),
        "terms_kept": int(np.count_nonzero(lasso.coef_)), "terms_total": int(Z.shape[1]),
        "terms": pd.DataFrame({"term": names, "coef": lasso.coef_}).query("coef != 0")
                   .sort_values("coef", key=np.abs, ascending=False),
    }
    return lasso.predict(Zn), lasso.predict(Z), info


def out_of_fold(kind, degree, X, y, feats):
    oof = np.zeros(len(y))
    for tr, va in pu.get_kfold().split(X):
        oof[va] = fit_predict(kind, degree, X[tr], y[tr], X[va], feats)[0]
    return oof


def write_predictions(path, preds):
    """Edit here if the sample submission needs extra columns (e.g. an id)."""
    pd.DataFrame({"y": preds}).to_csv(path, index=False)


def plot_curves(curves, var, chosen_model, chosen_degree, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for model, g in curves.groupby("model"):
        # Plot Validation MSE (Solid Line with Error Bars)
        p = ax.errorbar(g["degree"], g["val_mse"], yerr=g["val_se"], marker="o", capsize=3, label=f"{model} (Val CV)")
        # Plot Training MSE (Dashed Line, Same Color)
        ax.plot(g["degree"], g["train_mse"], marker="x", linestyle="--", color=p[0].get_color(), label=f"{model} (Train)")
        
    ax.axvline(chosen_degree, ls=":", c="gray", label=f"Chosen Degree ({chosen_degree})")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title(f"{var}: Train vs Validation MSE (Bias-Variance Tradeoff)")
    
    # Place legend outside the plot area
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def run(var, args):
    print(f"\n{'=' * 64}\n{var} ({args.roll})\n{'=' * 64}")
    X, y, X_test, feats = pu.load_data(args.roll, var, args.data_dir)
    print(f"train rows {len(y)}, test rows {len(X_test)}")

    ridge_degrees = pu.parse_degrees(args.ridge_degrees or pu.DEFAULT_RIDGE_DEGREES[var])
    lasso_degrees = pu.parse_degrees(args.lasso_degrees or pu.DEFAULT_LASSO_DEGREES[var])
    print(f"ridge degrees {ridge_degrees}\nlasso degrees {lasso_degrees}")

    curves = cv_table(X, y, ridge_degrees, lasso_degrees, args.jobs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out_dir / f"cv_results_{var}.csv", index=False)
    print("\n" + curves.round(4).to_string(index=False))

    # model + degree selection: best raw CV val, then the simplest degree within one SE of it
    best = curves.loc[curves["val_mse"].idxmin()]
    kind = best["model"]
    same = curves[curves["model"] == kind]
    degree = int(same[same["val_mse"] <= best["val_mse"] + best["val_se"]]["degree"].min())
    print(f"\nBest raw CV: {kind}, degree {int(best['degree'])}, Val MSE {best['val_mse']:.5g}")
    print(f">> Chosen: {kind}, degree {degree} (one-standard-error rule)")
    if int(best["degree"]) == int(same["degree"].max()):
        print("  [warn] best degree is the largest tried for this model: widen the degree list")

    # Evaluate Out-of-Fold (Honest predictive power)
    oof = out_of_fold(kind, degree, X, y, feats)
    print(f"\nOut-of-fold Val MSE {mean_squared_error(y, oof):.5g} | Val R2 {r2_score(y, oof):.5f}")

    # Fit final model on all data and evaluate full Training metrics
    preds, preds_train, info = fit_predict(kind, degree, X, y, X_test, feats)
    full_train_mse = mean_squared_error(y, preds_train)
    print(f"Total Train MSE     {full_train_mse:.5g} | Train R2 {r2_score(y, preds_train):.5f}")
    
    print(f"\nfinal alpha {info['alpha']:.3g}")
    if kind == "lasso":
        print(f"terms kept: {info['terms_kept']} of {info['terms_total']}")
        if info["alpha_at_grid_edge"]:
            print("  [warn] Lasso alpha hit the smallest value on its grid; lower eps in polyutils.make_lasso")
        info["terms"].to_csv(out_dir / f"terms_{var}.csv", index=False)
        print(info["terms"].head(12).round(3).to_string(index=False))

    write_predictions(out_dir / f"pred_{var}.csv", preds)
    print(f"\nsaved {out_dir / f'pred_{var}.csv'}")
    plot_curves(curves, var, kind, degree, out_dir / f"cv_curve_{var}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", required=True, help="e.g. BT2024072")
    ap.add_argument("--vars", nargs="+", default=["var1", "var2"], choices=["var1", "var2"])
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--ridge-degrees", default=None, help="e.g. 1-10 (default depends on the problem)")
    ap.add_argument("--lasso-degrees", default=None, help="e.g. 3-8 or 6,8,10")
    ap.add_argument("--jobs", type=int, default=-1, help="worker processes (-1 = all cores)")
    args = ap.parse_args()
    for var in args.vars:
        run(var, args)


if __name__ == "__main__":
    main()