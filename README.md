# Polynomial Regression: Regularization and Degree Optimization

**Machine Learning Assignment 1**

This repository contains the code, experiments, and final predictions for developing robust polynomial regression models to predict continuous target variables across two unique geological engineering scenarios. The project emphasizes strict cross-validation, feature scaling, and regularization to navigate the bias-variance tradeoff and prevent overfitting in high-degree polynomial expansions.

## Project Scenarios

- **Phase 1: Power Plant Steam Turbine Optimization (`var1`)**
  - **Objective:** Predict the Net Power Score based on 6 operational turbine parameters.
  - **Chosen Model:** **Lasso Regression (Degree 5)**. Given the high-dimensional feature space (461 terms), Lasso successfully enforced sparsity by shrinking approximately 75% of the interactive noise terms to zero, retaining only the 113 most predictive features.

- **Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)**
  - **Objective:** Predict the Thermal Anomaly Score based on 3D spatial coordinate offsets.
  - **Chosen Model:** **Ridge Regression (Degree 10)**. Ridge uniformly shrank weights to handle dense spatial interactions smoothly. The degree was selected using the **One-Standard-Error Rule** (stepping back from the absolute minimum Validation MSE at degree 12 to a simpler, equally viable degree 10 model).

## Repository Structure

- `train_and_predict.py`: The main execution script. Performs 5-fold cross-validation, calculates training/validation MSE, selects the optimal degree using the 1-SE rule, and generates final test predictions and learning curves.
- `polyutils.py`: Shared utility functions containing the data loaders, identical CV splitters, and the scikit-learn model pipelines.
- `exp1_baselines_and_scaling.py`: Experimental script comparing OLS (Ordinary Least Squares) baselines against regularized models to justify scaling and regularization choices.
- `exp2_regularization_comparison.py`: Experimental script directly comparing Ridge and Lasso performance and term-retention across both problem variants.
- `requirements.txt`: Python package dependencies.
- `report.pdf`: The final compiled academic report detailing the methodology, rationale, and bias-variance tradeoff analysis.
- `*.csv` & `*.png`: Output files including final predictions (`pred_var1.csv`, `pred_var2.csv`), experimental results, and generated bias-variance tradeoff plots.

## Methodology Highlights

1. **Strict Data Isolation:** All scaling and polynomial transformations are strictly fitted within the training folds of the cross-validation loop to prevent data leakage.
2. **Numerical Stability:** Inputs are transformed using `MinMaxScaler(-1, 1)` prior to polynomial expansion to prevent high-degree terms from causing numerical overflow.
3. **Generalization:** Final models are evaluated using out-of-fold validation metrics to provide an honest assessment of predictive power on unseen data.

## Setup and Execution

### 1. Create and activate a virtual environment

```bash
python -m venv env

# Windows
.\env\Scripts\activate

# macOS/Linux
source env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the final training and prediction pipeline

Ensure your dataset files, such as `<ROLL_NO>_train_var1.csv`, are located in the repository root directory.

```bash
python train_and_predict.py --roll <ROLL_NO>
```