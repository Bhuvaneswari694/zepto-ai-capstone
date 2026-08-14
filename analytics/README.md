# Module 2 - Analytics Pipeline (/analytics)

## Project Overview

This project analyzes the Titanic dataset in two stages:

1. Exploratory Data Analysis (EDA)
2. Machine Learning Modeling

The dataset is downloaded only once in `01_eda.py` using `sns.load_dataset()`.

After downloading, the raw dataset is immediately saved as `titanic.csv`.

The second script, `02_modeling.py`, does not download the dataset again. It simply reads `titanic.csv`.

This approach avoids unnecessary network requests and keeps the entire pipeline consistent.

---

## Requirements

Install all required libraries:

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Step 1: Run EDA

```bash
python 01_eda.py
```

This script:

- Loads the Titanic dataset
- Analyzes the data
- Handles missing values
- Creates charts
- Generates an EDA report

**Output files:**

- `titanic.csv`
- `titanic_cleaned_eda.csv`
- `eda_report.md`
- `charts/*.png`

---

### Step 2: Run the Modeling Pipeline

```bash
python 02_modeling.py
```

This script:

- Splits the dataset
- Preprocesses the data
- Trains machine-learning models
- Evaluates model performance
- Performs hyperparameter tuning
- Saves the final model

**Output files:**

- `modeling_report.md`
- `full_pipeline.joblib`
- Additional charts

---

## Project Files

| File | Description |
| --- | --- |
| `titanic.csv` | Raw Titanic dataset |
| `titanic_cleaned_eda.csv` | Cleaned dataset created during EDA |
| `eda_report.md` | EDA report |
| `modeling_report.md` | Modeling report |
| `charts/*.png` | All generated charts |
| `full_pipeline.joblib` | Saved machine-learning pipeline |

---

## EDA Tasks (Tasks 1-6)

### Task 1: Load and Profile the Dataset

- Load the Titanic dataset
- Display dataset information
- Identify missing values

### Task 2: Handle Missing Values

Rules used:

- Missing values below 5% → Remove rows
- Missing values between 5% and 30% → Fill missing values
- Missing values above 30% → Drop the column

### Task 3: Univariate Analysis

Analyze:

- Age
- Fare

Techniques used:

- Histogram
- Box plot
- Outlier detection

### Task 4: Bivariate Analysis

Analyze relationships between:

- Survival and sex
- Survival and passenger class

Techniques used:

- Boolean masking
- Correlation analysis
- Heatmap

### Task 5: Multivariate Analysis

Create a data story using multiple variables.

Charts created:

- Survival by class and sex
- Age by survival
- Fare by age and survival
- Survival by embarkation port
- Pair plot

### Task 6: Standardization Check

Apply Z-score standardization to:

- Age
- Fare

---

## Modeling Tasks (Tasks 7-15)

### Task 7: Train-Test Split

Split the data into:

- Training set (80%)
- Testing set (20%)

Stratification is used to maintain the original class distribution.

### Task 8: Data Preprocessing

Numeric features:

- Median imputation
- StandardScaler

Categorical features:

- Most frequent imputation
- OneHotEncoder

### Task 9: Train Classification Models

Models used:

- Logistic Regression
- Decision Tree
- Random Forest

### Task 10: Evaluate Models

Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Task 11: Handle Class Imbalance

Strategies compared:

- Baseline model
- Class weights
- SMOTE

### Task 12: Hyperparameter Tuning

Use GridSearchCV to find the best Random Forest model.

### Task 13: Regression Task

Use Linear Regression to predict passenger ticket fares.

Metrics used:

- MAE
- RMSE
- R²
- Adjusted R²

### Task 14: Compare Models

Compare all models and recommend the best one.

### Task 15: Save the Final Model

Save the complete machine-learning pipeline using:

```python
joblib.dump()
```

The saved model can be loaded later without retraining.

---

## Final Output

After running both scripts, the project generates:

- EDA report
- Modeling report
- Charts
- Cleaned dataset
- Saved machine-learning model
