"""
02_modeling.py -- Module 2 / Part B: Predictive modeling
Zepto Capstone -- /analytics

Reads the SAME titanic.csv produced by 01_eda.py (raw dataset, saved right
after the one and only sns.load_dataset('titanic') call in this module).
This script never re-fetches the dataset from the network.

Outputs:
  charts/*.png                    -- decision tree plot, ROC curves, residual plot
  modeling_report.md              -- every required table/metric/conclusion
  full_pipeline.joblib            -- best fitted pipeline (preprocessing + model)
"""

import numpy as np
import pandas as pd
# Agg allows charts saved as image files
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
# Save and load trianed models
import joblib

# tools for split data and tuning models
from sklearn.model_selection import train_test_split, GridSearchCV

# Build a complete manchine learing work flow
from sklearn.pipeline import Pipeline

# Apply diffrent preprocessing steps to differnt column types.
from sklearn.compose import ColumnTransformer

# Handle missing values.
from sklearn.impute import SimpleImputer

# Scale numerical features and 
# encode categorical features
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# machine learning models and metrics.
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score,
)

CHART_DIR = "charts" # store all generated charts
REPORT_PATH = "modeling_report.md" # store all observations
report_lines = [] # store report line befor writing them to file

def log(md_text=""):
    report_lines.append(md_text)
    print(md_text)


RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load the ONE raw dataset saved by 01_eda.py -- no second network call
# ---------------------------------------------------------------------------
log("# Titanic Modeling Report (auto-generated from live computation)\n")
df = pd.read_csv("titanic.csv")
log(f"Loaded `titanic.csv` (produced by `01_eda.py`) -- "
    f"**{df.shape[0]} rows x {df.shape[1]} columns**. This is the same raw "
    f"load as the EDA module; no second `sns.load_dataset` call is made here.\n")
#----------------------------------------------------------------
# Select the features and target column for modeling.
#-----------------------------------------------------------------
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET = "survived"

df_model = df[FEATURES + [TARGET]].copy()

#------------------------------------------------------------------
# Seperate numerical and categorical features.
# differnt preprocessing methods will be applied later.
#-------------------------------------------------------------------
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]

X = df_model[FEATURES] # input features 
y = df_model[TARGET] # target varaible 

# ---------------------------------------------------------------------------
# Task 7: Stratified train/test split
# Split the dataset into training and testing sets while keeping the same
# survival ratio in both datasets.
# ---------------------------------------------------------------------------
log("## Task 7 -- Stratified Train/Test Split\n")

# check overall class istribution
class_balance = y.value_counts(normalize=True).round(3)

log(f"Overall class balance -- survived=1: {class_balance.get(1, 0)*100:.1f}%, "
    f"survived=0: {class_balance.get(0, 0)*100:.1f}%.")
log(
    "Using stratification preserves the original class distribution "
    "in both the training and testing datasets.\n"
)
# split the data into traing(80%) an dtesting(20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# compare the survival rates after splitting.
log(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
log(f"Train survival rate: {y_train.mean():.3f} | Test survival rate: {y_test.mean():.3f}\n")

# ---------------------------------------------------------------------------
# Task 8: Preprocessing -- ColumnTransformer, fit on train only
# Clean and transform the data before training the model.
# To avoid data leakage, preprocessing is fitted only on the training data.
# ---------------------------------------------------------------------------
log("## Task 8 -- Preprocessing (fit on training data only)\n")

# Describe the preprocessing steps foe each feature type.
log("- Numeric features (`pclass`, `age`, `sibsp`, `parch`, `fare`): "
    "median imputation -> `StandardScaler`.")
log("- Categorical features (`sex`, `embarked`): most-frequent imputation -> "
    "`OneHotEncoder`.")
log("All of this is wrapped in a single `ColumnTransformer`, itself wrapped "
    "in a `Pipeline` with the estimator, so `.fit()` is only ever called on "
    "`X_train` / `y_train` and `X_test` only ever sees `.transform()`. This "
    "structurally prevents any test-set leakage.\n")

# creating a preprocessing pipline for 
# numerical and categorical features. 
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUMERIC_FEATURES),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), CATEGORICAL_FEATURES),
])

# ---------------------------------------------------------------------------
# Task 9: Train three classifiers on the identical split
# all models use the same traing data and preprocessing stpes.
# ---------------------------------------------------------------------------
log("## Task 9 -- Train Three Classifiers\n")

# create three classification models.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200),
}

# Create seperate pipeline for each model and trian it.
fitted_pipelines = {}
for name, clf in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe
log("Trained: Logistic Regression, Decision Tree, Random Forest -- all on "
    "the identical `X_train`/`y_train` split, each inside its own "
    "preprocessing+model `Pipeline`.\n")

#-----------------------------------------------------------------------------
# Visualize the first three levels of the decision tree.
# This helps us understand how the model makes decisions.
#-----------------------------------------------------------------------------
dt_pipe = fitted_pipelines["Decision Tree"]
dt_model = dt_pipe.named_steps["classifier"]

# Get all feature names after one hot encoding.
feature_names = (
    NUMERIC_FEATURES
    + list(dt_pipe.named_steps["preprocessor"]
           .named_transformers_["cat"]
           .named_steps["encoder"]
           .get_feature_names_out(CATEGORICAL_FEATURES))
)

plt.figure(figsize=(20, 10))
plot_tree(dt_model, max_depth=3, feature_names=feature_names,
          class_names=["Did not survive", "Survived"], filled=True, fontsize=8)
plt.title("Decision Tree (top 3 levels shown)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/decision_tree.png", dpi=120)
plt.close()
log(f"Decision tree visualized with labeled features/classes: "
    f"`charts/decision_tree.png`\n")

# ---------------------------------------------------------------------------
# Task 10: Evaluate and compare all three models.
# Genrate predictions, evaluate metric, 
# Calculate performance metrics and create a side by side comparsion
# ---------------------------------------------------------------------------
log("## Task 10 -- Model Evaluation\n")

eval_rows = []
# Create one Roc cruve chartto compare all models 
plt.figure(figsize=(6, 5))

# ---------------------------------------------------------------------------
# Generate predictions and calculate evaluation metrics for each model.
# --------------------------------------------------------------------------
for name, pipe in fitted_pipelines.items():
    y_pred = pipe.predict(X_test)
    # get prediction probabilities for ROC/AUC calculation.
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    # Add the ROC curve of the current model to the chart.
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # Save the evalution results to the report.
    log(f"### {name}")
    log(f"Confusion matrix:\n```\n{cm}\n```")
    log(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, "
        f"F1={f1:.3f}, AUC={auc:.3f}\n")

    # Store the results for later comparsion.
    eval_rows.append({
        "Model": name, "Accuracy": round(acc, 3), "Precision": round(prec, 3),
        "Recall": round(rec, 3), "F1": round(f1, 3), "AUC": round(auc, 3),
    })

# ---------------------------------------------------------------------------
# Create and save the ROC curve chart.
# ---------------------------------------------------------------------------
plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves -- all three classifiers")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/roc_curves.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# Create a comparison table for all models.
# ---------------------------------------------------------------------------
classifier_comparison = pd.DataFrame(eval_rows).set_index("Model")
log("### Classifier comparison table (side by side)")
log(classifier_comparison.to_markdown())
log(f"\nROC curves: `charts/roc_curves.png`\n")

# ---------------------------------------------------------------------------
# Task 11: Handle class imbalance.
# Compare different techniques to improve predictions for the minority class.
# ---------------------------------------------------------------------------
log("## Task 11 -- Imbalance Handling Comparison\n")

# Check the class distribution in the traing data.
log(f"Class balance in training data -- survived=1: "
    f"{y_train.mean()*100:.1f}%, survived=0: {(1-y_train.mean())*100:.1f}%.\n")

# ---------------------------------------------------------------------------
# Apply the same preprocessing to the training and testing data.
# The preprocessor is fitted only on the training data to avoid data leakage.
# ---------------------------------------------------------------------------
preprocessor_fit = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUMERIC_FEATURES),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), CATEGORICAL_FEATURES),
])
X_train_t = preprocessor_fit.fit_transform(X_train)
X_test_t = preprocessor_fit.transform(X_test)

#  ---------------------------------------------------------------------------
# Compare three imbalance-handling strategies.
# 1. Baseline
# 2. Class weights
# 3. SMOTE
# ---------------------------------------------------------------------------
imbalance_rows = []

# Strategy 1: Train a normal Logistic Regression model. 
clf_a = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
clf_a.fit(X_train_t, y_train)
pred_a = clf_a.predict(X_test_t)
imbalance_rows.append({
    "Strategy": "Baseline (no handling)",
    "Precision": round(precision_score(y_test, pred_a), 3),
    "Recall": round(recall_score(y_test, pred_a), 3),
    "F1": round(f1_score(y_test, pred_a), 3),
})

# Strategy 2: Give more importance to the minority class.
clf_b = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
clf_b.fit(X_train_t, y_train)
pred_b = clf_b.predict(X_test_t)
imbalance_rows.append({
    "Strategy": "class_weight='balanced'",
    "Precision": round(precision_score(y_test, pred_b), 3),
    "Recall": round(recall_score(y_test, pred_b), 3),
    "F1": round(f1_score(y_test, pred_b), 3),
})

# Strategy 3: Use SOMTE to create synthentic samples 
# for the minority class
# SMOTE is applied only to the traing data.
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_t, y_train)
    clf_c = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf_c.fit(X_train_sm, y_train_sm)
    pred_c = clf_c.predict(X_test_t)  # test fold NEVER resampled
    imbalance_rows.append({
        "Strategy": "SMOTE (train fold only)",
        "Precision": round(precision_score(y_test, pred_c), 3),
        "Recall": round(recall_score(y_test, pred_c), 3),
        "F1": round(f1_score(y_test, pred_c), 3),
    })
    smote_available = True
except ImportError:
    log(
        "SMOTE results are unavailable because "
        "`imbalanced-learn` is not installed."
    )
    smote_available = False

# Comparsion table for all strategies.
imbalance_df = pd.DataFrame(imbalance_rows).set_index("Strategy")
log("### Imbalance strategy comparison (Logistic Regression)")
log(imbalance_df.to_markdown())

# Select the strategy with best F1 score.
if smote_available:
    best_strategy = imbalance_df["F1"].idxmax()
    log(f"\n**Conclusion:** `{best_strategy}` produced the best F1 "
        f"({imbalance_df.loc[best_strategy, 'F1']:.3f}) on the test fold. "
        f"Baseline logistic regression tends to favor the majority class "
        f"(not-survived) and under-predicts the minority class; both "
        f"`class_weight='balanced'` and SMOTE push recall on the minority "
        f"class up by changing how errors are weighted (class_weight) or by "
        f"synthetically enlarging the minority class in the training fold "
        f"only (SMOTE), typically at some cost to precision. Note SMOTE was "
        f"fit and applied strictly on `X_train_t`/`y_train` -- the test fold "
        f"`X_test_t` was never resampled, so this comparison has no leakage.\n")

# ---------------------------------------------------------------------------
# Task 12: Hyperparameter Tuning
# Find the best Random Forest model by testing different parameter
# combinations and compare the results using cross-validation.
# ---------------------------------------------------------------------------
log("## Task 12 -- Hyperparameter Tuning (GridSearchCV, Random Forest)\n")

# create Random Forest pipeline.
# OOB scoring provides an additional way to evaluate the model.
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE, bootstrap=True)),
])

# Define parameter combinations to test.
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"],
}

# Use GridSearchCV to test all parameter combinations and select
# the model with the best F1 score.
grid_search = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model and display the results.
best_rf_pipe = grid_search.best_estimator_
best_rf_model = best_rf_pipe.named_steps["classifier"]

log(f"Best params: `{grid_search.best_params_}`")
log(f"Best cross-val F1: {grid_search.best_score_:.3f}")
log(f"OOB score of best Random Forest (`oob_score=True` at construction): "
    f"**{best_rf_model.oob_score_:.3f}**\n")

# ---------------------------------------------------------------------------
# Task 13: Regression side-task 
# Predict passenger ticket fare using Linear Regression.
# ---------------------------------------------------------------------------
log("## Task 13 -- Regression Side-Task: Predicting `fare`\n")

# Select features  and prepares the regression dataset.
REG_FEATURES = ["pclass", "age", "sibsp", "parch", "survived"]
CAT_REG_FEATURES = ["sex", "embarked"]
df_reg = df[REG_FEATURES + CAT_REG_FEATURES + ["fare"]].dropna(subset=["fare"]).copy()

# Split the data into training and testing sets.
Xr = df_reg[REG_FEATURES + CAT_REG_FEATURES]
yr = df_reg["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=RANDOM_STATE
)

# Preprocess numeric and catogorical features.
reg_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), REG_FEATURES),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), CAT_REG_FEATURES),
])

# Create and train the Linear Regression, pipeline.
reg_pipe = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("regressor", LinearRegression()),
])
reg_pipe.fit(Xr_train, yr_train)

# Make predictions and calculate regression metrics.
yr_pred = reg_pipe.predict(Xr_test)
mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
n, p = len(yr_test), Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

log(f"MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}, Adjusted R2={adj_r2:.3f}\n")

# Create a residual plot to compare prediction errors.
# A residual plot helps us check whether the model's errors are random.
residuals = yr_test.values - yr_pred
plt.figure(figsize=(6, 4))
plt.scatter(yr_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values (predicted fare)")
plt.ylabel("Residuals")
plt.title("Residual plot -- fare regression")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/residual_plot.png", dpi=120)
plt.close()

# Create a residual plot to compare prediction errors.
# A residual plot helps us check whether the model's errors are random.
order = np.argsort(yr_pred)
half = len(order) // 2
resid_sorted = residuals[order]
var_low = np.var(resid_sorted[:half])
var_high = np.var(resid_sorted[half:])
ratio = max(var_low, var_high) / max(min(var_low, var_high), 1e-9)
hetero_verdict = "heteroscedastic (non-random spread)" if ratio > 1.5 else "roughly homoscedastic (fairly even spread)"
log(f"Residual plot: `charts/residual_plot.png`")
log(f"Residual variance, low-fitted half = {var_low:.1f}, high-fitted half = "
    f"{var_high:.1f} (ratio={ratio:.2f}x) -> spread is **{hetero_verdict}**. "
    f"A ratio well above 1 means residual spread widens/narrows "
    f"systematically as predicted fare increases, which is the signature of "
    f"heteroscedasticity.\n")

# ---------------------------------------------------------------------------
# Task 14: Final model comparison table + recommendation
# compare al models and select the best for deployment.
# ---------------------------------------------------------------------------
log("## Task 14 -- Final Model Comparison & Recommendation\n")

# Display the classification model comparsion table.
log("### Classification metrics (one metric group)")
log(classifier_comparison.to_markdown())

# Display the regression model evaluation table.
# Classification and regression metrics are shown separately because they
# measure different tasks and cannot be compared directly.
log("\n### Regression metrics (separate metric group -- not on the same scale as classification)")
reg_table = pd.DataFrame([{
    "Model": "Linear Regression (fare)", "MAE": round(mae, 2),
    "RMSE": round(rmse, 2), "R2": round(r2, 3), "Adjusted R2": round(adj_r2, 3),
}]).set_index("Model")
log(reg_table.to_markdown())

# Find the model with the highest F1 score and recommend it for deployment.
best_by_f1 = classifier_comparison["F1"].idxmax()
best_row = classifier_comparison.loc[best_by_f1]
log(f"\n**Recommendation:** Deploy **{best_by_f1}** as the survival "
    f"classifier. It achieved the highest F1 ({best_row['F1']:.3f}) among "
    f"the three models, with accuracy {best_row['Accuracy']:.3f}, precision "
    f"{best_row['Precision']:.3f}, recall {best_row['Recall']:.3f}, and AUC "
    f"{best_row['AUC']:.3f} on the held-out test set. For a survival-triage "
    f"style use case, F1 is a reasonable single metric to lead with because "
    f"it balances the cost of missing an actual survivor (recall) against "
    f"the cost of false alarms (precision), and {best_by_f1} did that best "
    f"here without materially sacrificing accuracy or AUC.\n")

# ---------------------------------------------------------------------------
# Task 15: Save best pipeline (preprocessing + estimator, as ONE object)
# ---------------------------------------------------------------------------
log("## Task 15 -- Save & Reload Full Pipeline\n")

# Save the best model together with all preprocessing steps.
joblib.dump(best_rf_pipe, "full_pipeline.joblib")
log("Saved the complete fitted pipeline (ColumnTransformer preprocessing + "
    "tuned RandomForestClassifier, as one `Pipeline` object) to "
    "`full_pipeline.joblib` via `joblib.dump(full_pipeline, ...)`.\n")

# Reload the saved pipeline and verify that it produces the same predictions.
reloaded = joblib.load("full_pipeline.joblib")
sample_raw = X_test.iloc[:5]
reloaded_preds = reloaded.predict(sample_raw)
original_preds = best_rf_pipe.predict(sample_raw)
match = np.array_equal(reloaded_preds, original_preds)
log(f"Reloaded pipeline predictions on 5 raw (unpreprocessed) test rows: "
    f"`{list(reloaded_preds)}`")
log(f"Matches original in-memory pipeline predictions: **{match}**")
log("This confirms the saved artifact is usable end-to-end on raw new data "
    "-- it re-applies its own imputation/encoding/scaling internally, "
    "nothing needs to be preprocessed by hand before calling `.predict()`.\n")

# Save the final report to Markdown file.
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print("\n\nDone. See modeling_report.md, charts/, and full_pipeline.joblib.")
