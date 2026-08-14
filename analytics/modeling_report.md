# Titanic Modeling Report (auto-generated from live computation)

Loaded `titanic.csv` (produced by `01_eda.py`) -- **891 rows x 15 columns**. This is the same raw load as the EDA module; no second `sns.load_dataset` call is made here.

## Task 7 -- Stratified Train/Test Split

Overall class balance -- survived=1: 38.4%, survived=0: 61.6%.
Using stratification preserves the original class distribution in both the training and testing datasets.

Train size: 712, Test size: 179
Train survival rate: 0.383 | Test survival rate: 0.385

## Task 8 -- Preprocessing (fit on training data only)

- Numeric features (`pclass`, `age`, `sibsp`, `parch`, `fare`): median imputation -> `StandardScaler`.
- Categorical features (`sex`, `embarked`): most-frequent imputation -> `OneHotEncoder`.
All of this is wrapped in a single `ColumnTransformer`, itself wrapped in a `Pipeline` with the estimator, so `.fit()` is only ever called on `X_train` / `y_train` and `X_test` only ever sees `.transform()`. This structurally prevents any test-set leakage.

## Task 9 -- Train Three Classifiers

Trained: Logistic Regression, Decision Tree, Random Forest -- all on the identical `X_train`/`y_train` split, each inside its own preprocessing+model `Pipeline`.

Decision tree visualized with labeled features/classes: `charts/decision_tree.png`

## Task 10 -- Model Evaluation

### Logistic Regression
Confusion matrix:
```
[[98 12]
 [23 46]]
```
Accuracy=0.804, Precision=0.793, Recall=0.667, F1=0.724, AUC=0.844

### Decision Tree
Confusion matrix:
```
[[97 13]
 [20 49]]
```
Accuracy=0.816, Precision=0.790, Recall=0.710, F1=0.748, AUC=0.790

### Random Forest
Confusion matrix:
```
[[98 12]
 [21 48]]
```
Accuracy=0.816, Precision=0.800, Recall=0.696, F1=0.744, AUC=0.830

### Classifier comparison table (side by side)
| Model               |   Accuracy |   Precision |   Recall |    F1 |   AUC |
|:--------------------|-----------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.804 |       0.793 |    0.667 | 0.724 | 0.844 |
| Decision Tree       |      0.816 |       0.79  |    0.71  | 0.748 | 0.79  |
| Random Forest       |      0.816 |       0.8   |    0.696 | 0.744 | 0.83  |

ROC curves: `charts/roc_curves.png`

## Task 11 -- Imbalance Handling Comparison

Class balance in training data -- survived=1: 38.3%, survived=0: 61.7%.

### Imbalance strategy comparison (Logistic Regression)
| Strategy                |   Precision |   Recall |    F1 |
|:------------------------|------------:|---------:|------:|
| Baseline (no handling)  |       0.793 |    0.667 | 0.724 |
| class_weight='balanced' |       0.73  |    0.783 | 0.755 |
| SMOTE (train fold only) |       0.74  |    0.783 | 0.761 |

**Conclusion:** `SMOTE (train fold only)` produced the best F1 (0.761) on the test fold. Baseline logistic regression tends to favor the majority class (not-survived) and under-predicts the minority class; both `class_weight='balanced'` and SMOTE push recall on the minority class up by changing how errors are weighted (class_weight) or by synthetically enlarging the minority class in the training fold only (SMOTE), typically at some cost to precision. Note SMOTE was fit and applied strictly on `X_train_t`/`y_train` -- the test fold `X_test_t` was never resampled, so this comparison has no leakage.

## Task 12 -- Hyperparameter Tuning (GridSearchCV, Random Forest)

Best params: `{'classifier__max_depth': 5, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 100}`
Best cross-val F1: 0.746
OOB score of best Random Forest (`oob_score=True` at construction): **0.827**

## Task 13 -- Regression Side-Task: Predicting `fare`

MAE=20.90, RMSE=30.53, R2=0.398, Adjusted R2=0.373

Residual plot: `charts/residual_plot.png`
Residual variance, low-fitted half = 148.4, high-fitted half = 1454.0 (ratio=9.80x) -> spread is **heteroscedastic (non-random spread)**. A ratio well above 1 means residual spread widens/narrows systematically as predicted fare increases, which is the signature of heteroscedasticity.

## Task 14 -- Final Model Comparison & Recommendation

### Classification metrics (one metric group)
| Model               |   Accuracy |   Precision |   Recall |    F1 |   AUC |
|:--------------------|-----------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.804 |       0.793 |    0.667 | 0.724 | 0.844 |
| Decision Tree       |      0.816 |       0.79  |    0.71  | 0.748 | 0.79  |
| Random Forest       |      0.816 |       0.8   |    0.696 | 0.744 | 0.83  |

### Regression metrics (separate metric group -- not on the same scale as classification)
| Model                    |   MAE |   RMSE |    R2 |   Adjusted R2 |
|:-------------------------|------:|-------:|------:|--------------:|
| Linear Regression (fare) |  20.9 |  30.53 | 0.398 |         0.373 |

**Recommendation:** Deploy **Decision Tree** as the survival classifier. It achieved the highest F1 (0.748) among the three models, with accuracy 0.816, precision 0.790, recall 0.710, and AUC 0.790 on the held-out test set. For a survival-triage style use case, F1 is a reasonable single metric to lead with because it balances the cost of missing an actual survivor (recall) against the cost of false alarms (precision), and Decision Tree did that best here without materially sacrificing accuracy or AUC.

## Task 15 -- Save & Reload Full Pipeline

Saved the complete fitted pipeline (ColumnTransformer preprocessing + tuned RandomForestClassifier, as one `Pipeline` object) to `full_pipeline.joblib` via `joblib.dump(full_pipeline, ...)`.

Reloaded pipeline predictions on 5 raw (unpreprocessed) test rows: `[np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(1)]`
Matches original in-memory pipeline predictions: **True**
This confirms the saved artifact is usable end-to-end on raw new data -- it re-applies its own imputation/encoding/scaling internally, nothing needs to be preprocessed by hand before calling `.predict()`.
