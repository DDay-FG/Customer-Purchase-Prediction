# Customer Purchase Prediction

Predicts whether a customer will make a purchase, using the Kaggle dataset
`rabieelkharoua/predict-customer-purchase-behavior-dataset` (binary target
`PurchaseStatus`). The pipeline screens features with Weight of Evidence /
Information Value, trains a logistic regression and an XGBoost classifier on
the surviving features, evaluates both with held-out and cross-validated
ROC-AUC plus a profit-based cost model, explains both with SHAP, and exports
the fitted models to a Streamlit app for interactive scoring. Built as the
final project for FINAN 6520 (Financial Programming in Python) at the
University of Utah by Group 5: Rachel Frandsen, Julio Cruz, Darrell Day,
Nasandelger Namkhai.

## Pipeline

`CustomerPurchaseBehavior.py` runs the following stages top to bottom and
writes all artifacts to a timestamped `output_<YYYYMMDD_HHMMSS>/` directory.

1. Data download. Authenticates with the Kaggle API, downloads the dataset
   into `kaggle_dataset/`, and loads `customer_purchase_data.csv`.
2. Data quality report. Records dataset shape, missing values, target
   distribution, and the class-imbalance ratio to `data_quality_report.txt`.
3. Correlation analysis. Saves a lower-triangle correlation heatmap of the
   numeric features.
4. WOE/IV feature screening. `WOE.py` (adapted from Sundar Krishnan's
   WOE-and-IV notebook, https://github.com/Sundar0989/WOE-and-IV) bins each
   feature: Spearman-guided monotonic quantile binning for continuous
   variables, per-category binning for categoricals, and missing values as a
   separate bin. It computes Weight of Evidence and Information Value per
   feature, exports the IV table to `IV_analysis.xlsx`, and the low-IV
   features `Gender` and `ProductCategory` are dropped.
5. Split and scale. 70/30 stratified train/test split (`random_state=42`)
   followed by `StandardScaler`.
6. Logistic regression. `LogisticRegression(solver='lbfgs', max_iter=1000,
   class_weight='balanced')`, scored with 5-fold `StratifiedKFold`
   cross-validated ROC-AUC. Coefficients and odds ratios are exported to
   `model_coefficients.xlsx` with a bar-chart rendering.
7. XGBoost. `xgb.train` with `binary:logistic`, `max_depth=4`,
   `learning_rate=0.1`, row/column subsampling, L1/L2 regularization, and
   early stopping, plus a 5-fold stratified `xgb.cv` run.
8. Evaluation. For each model: ROC curve, confusion matrix, classification
   report, and a profit estimate using example unit economics (false
   positive -$10 marketing cost, false negative -$50 missed sale, true
   positive +$100 conversion).
9. SHAP explanations. `LinearExplainer` for the logistic regression,
   `TreeExplainer` for XGBoost; summary, bar, and waterfall plots for each.
10. Comparison and export. Writes a model comparison report (recommending
    XGBoost only when its test AUC exceeds the logistic regression's by more
    than 0.05, otherwise favoring logistic regression for interpretability),
    serializes both models plus the scaler and feature metadata to
    `streamlit_assets/`, scores the full dataset with both models and a
    probability-average ensemble, and writes an executive summary.

## Results

Metrics are regenerated on every run and written to the output directory:
per-model ROC-AUC on the held-out 30%, cross-validation AUC, classification
reports, confusion matrices, and the profit estimates described above. No
benchmark numbers are recorded in the repository; see
`model_comparison_report.txt` and `project_summary_enhanced.txt` from your
run for the figures. One caveat as implemented: the XGBoost cross-validation
AUC written to the summary files is a hard-coded placeholder (0.85); the
actual `xgb.cv` result is printed to the console during training.

## How to run

Requires Python 3.8+ (developed on 3.11) and a Kaggle account.

Kaggle credentials:

1. On kaggle.com go to Settings, then API, then Create New Token. This
   downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (macOS/Linux) or
   `C:\Users\<user>\.kaggle\kaggle.json` (Windows).
3. On Unix systems: `chmod 600 ~/.kaggle/kaggle.json`.

Install and train:

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python CustomerPurchaseBehavior.py
```

Run the scoring app. `app.py` loads models from
`output_latest/streamlit_assets/`, while training writes to a timestamped
directory, so copy or link the run you want to serve first:

```
cp -r output_<timestamp> output_latest    # or: ln -s output_<timestamp> output_latest
streamlit run app.py
```

The app scores a customer with both models plus their probability average,
shows per-feature contributions (logistic coefficients) and gain-based
feature importance (XGBoost), and maps the ensemble probability to a
marketing recommendation.

Run the tests:

```
pytest test_models.py -v
```

Note: `test_models.py` imports `CustomerPurchaseBehavior`, which executes
the full pipeline at import time, so the test run also requires Kaggle
credentials and network access. CI therefore runs only a syntax and
undefined-name lint (`ruff check --select E9,F`).

## Repository layout

```
CustomerPurchaseBehavior.py   End-to-end training pipeline: download, WOE/IV
                              screening, both models, evaluation, SHAP, exports
WOE.py                        Weight of Evidence / Information Value binning module
app.py                        Streamlit scoring interface
test_models.py                pytest suite: WOE/IV, model training, business
                              metrics, edge cases, integration
requirements.txt              Pinned dependencies
.github/workflows/ci.yml      Lint on push (ruff E9/F)
LICENSE                       MIT
```

Generated at runtime and gitignored: `kaggle_dataset/`,
`output_<timestamp>/`, `output_latest/`.

## License

MIT. See `LICENSE`.

---
Provenance: written and published Aug 2025 under my University of Utah student account (https://github.com/dday-source/Customer-Purchase-Prediction). Republished here with full commit history; the original remains up.
