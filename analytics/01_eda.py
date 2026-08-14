"""
01_eda.py -- Module 2 / Part A: Profiling, cleaning, and the data story
Zepto Capstone -- /analytics

Run once, in an environment with internet access on first run (Seaborn will
cache the dataset locally after that). This script:
  1. Loads titanic exactly once via sns.load_dataset('titanic')
  2. Immediately saves the RAW dataframe to titanic.csv (offline fallback --
     used by 02_modeling.py so the dataset is never re-fetched from the network)
  3. Profiles, cleans, and explores the data, saving charts to charts/
  4. Writes every required written interpretation to eda_report.md, computed
     programmatically from the real, measured numbers (never hardcoded)

Outputs:
  titanic.csv          -- raw offline fallback (read by 02_modeling.py)
  charts/*.png          -- all EDA charts
  eda_report.md          -- profiling stats + all written interpretations
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

# "Agg" allows charts to be saved as image files even in
# environments that do not support graphical displays.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# All charts will be stored inside this folder.
CHART_DIR = "charts"

# All written observations will be stored in this file.
REPORT_PATH = "eda_report.md"

# Create the charts folder if it does not already exist.
os.makedirs(CHART_DIR, exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

#To store Markdown files 
report_lines = []


def log(md_text=""):
    """Append a line to the running markdown report, print, save to eda_report.md"""
    report_lines.append(md_text)
    print(md_text)


# ---------------------------------------------------------------------------
# Task 1: Load once, profile, save raw offline fallback
# ---------------------------------------------------------------------------
log("# Titanic EDA Report (auto-generated from live computation)\n")
log("## Task 1 -- Load & Profile\n")

#Load the Titanic dataset.
df_raw = sns.load_dataset("titanic")

# Immediatliy save this file so 02_modeling.py file uses this 
# csv file no need to dowload it again
df_raw.to_csv("titanic.csv", index=False)
log(f"- Raw dataset loaded via `sns.load_dataset('titanic')` and saved to "
    f"`titanic.csv` immediately after loading (one-time network/cache hit).")
log(f"- Shape: **{df_raw.shape[0]} rows x {df_raw.shape[1]} columns**\n")

# Display dataset information
log("### df.info()")
import io
# Create a temporary text buffer to store the output of df.info().
buf = io.StringIO()

# Save the df.info() output inside the buffer instead of printing it.
df_raw.info(buf=buf)

# Retrieve the stored text and add it to the report.
log("```\n" + buf.getvalue() + "```")

# Display statistical info
log("### df.describe()")
log(df_raw.describe(include="all").to_markdown())

# Identyfy missing values 
# count missing values in each column, keep them, print % for every col
missing_pct = (df_raw.isna().sum() / len(df_raw) * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
log("\n### Missing values (columns with any missing data)")
for col, pct in missing_pct.items():
    log(f"- `{col}`: {pct}% missing ({df_raw[col].isna().sum()} rows)")

# ---------------------------------------------------------------------------
# Task 2: Missing-value handling (threshold rule)
#   < 5%   -> drop those rows
#   5-30%  -> impute (median)
#   very high (unreliable to impute) -> drop column OR encode "missing" category
# ---------------------------------------------------------------------------
log("\n## Task 2 -- Missing Value Handling\n")

# Create a copy so the original dataset remains unchanged.
df = df_raw.copy()

# Store all cleaning decisions for the report.
cleaning_notes = []

# Check every column that contains missing values.
for col, pct in missing_pct.items():

    # If missing values are between 5% and 30% in a numeric column,
    # replace them with the median.
    if pct >= 5 and pct < 30 and pd.api.types.is_numeric_dtype(df[col]):

        median_val = df[col].median()

        df[col] = df[col].fillna(median_val)

        note = (
            f"- `{col}`: {pct}% missing -> "
            f"filled with median ({median_val:.2f})."
        )

        cleaning_notes.append(note)

    # If more than 30% of the values are missing,
    # remove the entire column.
    elif pct >= 30:

        df = df.drop(columns=[col])

        note = (
            f"- `{col}`: {pct}% missing -> "
            f"column dropped."
        )

        cleaning_notes.append(note)

    # If less than 5% of the values are missing,
    # remove those rows.
    elif pct < 5:

        # Count rows before cleaning.
        before = len(df)

        # Remove rows with missing values.
        df = df[df[col].notna()]

        # Count rows after cleaning.
        after = len(df)

        note = (
            f"- `{col}`: {pct}% missing -> "
            f"{before - after} rows removed."
        )

        cleaning_notes.append(note)

    # If the column is categorical,
    # replace missing values with the most common value.
    else:

        mode_val = df[col].mode(dropna=True)[0]

        df[col] = df[col].fillna(mode_val)

        note = (
            f"- `{col}`: {pct}% missing -> "
            f"filled with mode ({mode_val})."
        )

        cleaning_notes.append(note)

# Add all cleaning notes to the report.
for note in cleaning_notes:

    log(note)

# Display the dataset shape after cleaning.
log(
    f"\nShape after cleaning: "
    f"**{df.shape[0]} rows x {df.shape[1]} columns**"
)
# ---------------------------------------------------------------------------
# Task 3: Univariate Analysis - Age and Fare
# ---------------------------------------------------------------------------

log("\n## Task 3 -- Univariate Analysis: age & fare\n")

# Use the IQR method to find outliers.
def iqr_outlier_count(series):

    # Calculate the first and third quartiles, interquartile range,
    # upper and lower limits
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Select values outside the limits.
    outliers = series[
        (series < lower) | (series > upper)
    ]

    return len(outliers), lower, upper


# Analyze both age and fare.
for col in ["age", "fare"]:

    # Create histogram and box plot.
    # Box plot helps identify outliers.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(
        df[col],
        bins=30,
        color="#4C72B0",
        edgecolor="white"
    )
    axes[0].set_title(f"Histogram of {col}")
    axes[1].boxplot(df[col], vert=True)
    axes[1].set_title(f"Box plot of {col}")

    plt.tight_layout()

    # Save the chart.
    plt.savefig(
        f"{CHART_DIR}/univariate_{col}.png",
        dpi=120
    )

    plt.close()

    # Calculate the number of outliers.
    n_out, lower, upper = iqr_outlier_count(df[col])

    # Add the results to the report.
    log(
        f"- `{col}`: IQR outlier bounds = "
        f"[{lower:.2f}, {upper:.2f}] -> "
        f"**{n_out} outliers** "
        f"({n_out / len(df) * 100:.1f}% of rows). "
        f"Chart: `charts/univariate_{col}.png`"
    )

# Calculate mean, median, and mode for fare.
fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

# Determine the shape of the distribution.
if fare_mean > fare_median > fare_mode:
    skew_desc = (
        "right-skewed "
        "(mean > median > mode)"
    )

elif fare_mean < fare_median < fare_mode:
    skew_desc = (
        "left-skewed "
        "(mean < median < mode)"
    )

else:
    skew_desc = ("roughly symmetric")

# Save the interpretation.
log(
    f"\n- `fare`: mean = {fare_mean:.2f}, "
    f"median = {fare_median:.2f}, "
    f"mode = {fare_mode:.2f} -> "
    f"distribution is **{skew_desc}**."
)
# ---------------------------------------------------------------------------
# Task 4: Bivariate Analysis - Compare survival rates and study correlations.
# 1. Which groups survived more often?
# 2. Which variables have the strongest relationships?
# ---------------------------------------------------------------------------

log("\n## Task 4 -- Bivariate Analysis\n")

# Compare survival rates between female and male passengers.
survival_by_sex = (
    df[df["sex"] == "female"]["survived"].mean(),
    df[df["sex"] == "male"]["survived"].mean()
)
log("### Survival rate by sex (boolean masking)")
for sex_val in ["female", "male"]:
    rate = df.loc[df["sex"] == sex_val, "survived"].mean()
    log(f"- {sex_val}: {rate:.3f} ({rate*100:.1f}%)")


# Compare survival rates across passenger classes.
log("\n### Survival rate by pclass (boolean masking)")
for pc in sorted(df["pclass"].unique()):
    rate = df.loc[df["pclass"] == pc, "survived"].mean()
    log(f"- pclass {pc}: {rate:.3f} ({rate*100:.1f}%)")

# Combine gender and passenger class to find the survival rate for each group.
log("\n### Survival rate by sex AND pclass (boolean masking with &)")

combo_rates = {}

for sex_val in ["female", "male"]:
    for pc in sorted(df["pclass"].unique()):
        mask = (
            (df["sex"] == sex_val)
            & (df["pclass"] == pc)
        )
        rate = df.loc[mask, "survived"].mean()
        combo_rates[(sex_val, pc)] = rate
        log(
            f"- {sex_val}, pclass {pc}: "
            f"{rate:.3f} ({rate*100:.1f}%)"
        )


# Create a correlation matrix for selected numerical columns.
corr_cols = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]
corr_matrix = df[corr_cols].corr()

# Create and save the correlation heatmap.
plt.figure(figsize=(7, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation matrix (6 numeric columns)")

plt.tight_layout()

plt.savefig(
    f"{CHART_DIR}/correlation_heatmap.png",
    dpi=120
)

plt.close()


# Find the two strongest correlations.
# Sort pairs, take first 2 results, 
pairs = []

for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        c1, c2 = corr_cols[i], corr_cols[j]
        pairs.append(
            (c1, c2, corr_matrix.loc[c1, c2])
        )
pairs_sorted = sorted(
    pairs,
    key=lambda x: abs(x[2]),
    reverse=True
)
top2 = pairs_sorted[:2]

log(
    f"\n### Correlation heatmap: "
    f"`charts/correlation_heatmap.png`"
)
log("\n**Two strongest correlations (by absolute value):**")

for c1, c2, val in top2:
    direction = (
        "positive" if val > 0 else "negative"
    )

    log(
        f"- `{c1}` vs `{c2}`: "
        f"r = {val:.3f} "
        f"({direction} correlation)"
    )


# Summarize the strongest relationships.
log(
    f"\nInterpretation: the strongest relationship is between "
    f"`{top2[0][0]}` and `{top2[0][1]}` "
    f"(r={top2[0][2]:.3f}), followed by "
    f"`{top2[1][0]}` and `{top2[1][1]}` "
    f"(r={top2[1][2]:.3f})."
)

# ---------------------------------------------------------------------------
# Task 5: Multivariate Data Story
# Create 5 charts to understand how multiple variables affect survival.
# Each chart is chosen because it answers a different question.
# ---------------------------------------------------------------------------

log("\n## Task 5 -- Multivariate Data Story (4+ charts)\n")

# ---------------------------------------------------------------------------
# Chart 1: Grouped bar chart
# Why this plot?
# Bar charts are useful for comparing values between groups.
# Here, we compare survival rates across gender and passenger class.
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 4))

grp = df.groupby(
    ["pclass", "sex"]
)["survived"].mean().unstack()

grp.plot(kind="bar", ax=plt.gca())
plt.ylabel("Survival rate")
plt.title("Survival rate by class and sex")
plt.tight_layout()
plt.savefig(
    f"{CHART_DIR}/story_1_survival_by_class_sex.png",
    dpi=120
)
plt.close()
best_combo = max(combo_rates, key=combo_rates.get)
worst_combo = min(combo_rates, key=combo_rates.get)

log(
    f"**Chart 1 -- Survival by class & sex** "
    f"(`charts/story_1_survival_by_class_sex.png`): "
    f"{best_combo[0]} passengers in pclass "
    f"{best_combo[1]} had the highest survival rate "
    f"({combo_rates[best_combo]*100:.1f}%), while "
    f"{worst_combo[0]} passengers in pclass "
    f"{worst_combo[1]} had the lowest "
    f"({combo_rates[worst_combo]*100:.1f}%)."
)
# ---------------------------------------------------------------------------
# Chart 2: Box plot
# Why this plot?
# Box plots are useful for comparing distributions and identifying
# differences between groups.
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 4))

sns.boxplot(
    data=df,
    x="survived",
    y="age"
)
plt.title("Age distribution by survival outcome")
plt.tight_layout()
plt.savefig(
    f"{CHART_DIR}/story_2_age_by_survival.png",
    dpi=120
)
plt.close()

age_surv = df.loc[
    df["survived"] == 1,
    "age"
].median()

age_died = df.loc[
    df["survived"] == 0,
    "age"
].median()

log(
    f"\n**Chart 2 -- Age by survival** "
    f"(`charts/story_2_age_by_survival.png`): "
    f"median age of survivors is {age_surv:.1f} "
    f"vs {age_died:.1f} for non-survivors."
)
# ---------------------------------------------------------------------------
# Chart 3: Scatter plot
# Why this plot?
# Scatter plots help us identify relationships between two
# continuous variables.
# ---------------------------------------------------------------------------

plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.6
)
plt.title("Fare vs age, coloured by survival")
plt.tight_layout()
plt.savefig(
    f"{CHART_DIR}/story_3_fare_age_survival.png",
    dpi=120
)
plt.close()

log(
    f"\n**Chart 3 -- Fare vs age by survival** "
    f"(`charts/story_3_fare_age_survival.png`)."
)
# ---------------------------------------------------------------------------
# Chart 4: Bar plot
# Why this plot?
# Bar plots make it easy to compare average values across categories.
# ---------------------------------------------------------------------------

plt.figure(figsize=(6, 4))

sns.barplot(
    data=df,
    x="embarked",
    y="survived",
    errorbar=None
)
plt.title("Survival rate by embarkation point")
plt.tight_layout()
plt.savefig(
    f"{CHART_DIR}/story_4_survival_by_embarked.png",
    dpi=120
)
plt.close()

embark_rates = df.groupby(
    "embarked"
)["survived"].mean()

best_port = embark_rates.idxmax()

log(
    f"\n**Chart 4 -- Survival by embarkation port** "
    f"(`charts/story_4_survival_by_embarked.png`): "
    f"'{best_port}' had the highest survival rate."
)
# ---------------------------------------------------------------------------
# Chart 5: Pair plot
# Why this plot?
# Pair plots compare multiple variables at the same time.
# ---------------------------------------------------------------------------

pairplot_fig = sns.pairplot(
    df[
        ["survived", "pclass", "age", "fare"]
    ].dropna(),
    hue="survived",
    diag_kind="hist",
    plot_kws={"alpha": 0.5}
)

pairplot_fig.savefig(
    f"{CHART_DIR}/story_5_pairplot.png",
    dpi=120
)

plt.close("all")

log(
    f"\n**Chart 5 -- Pair plot** "
    f"(`charts/story_5_pairplot.png`)."
)
# ---------------------------------------------------------------------------
# Task 6: Exploratory Standardization Check
# This step checks whether age and fare are correctly standardized.
# This transformed data is used only for EDA and not for model training.
# ---------------------------------------------------------------------------

log("\n## Task 6 -- Exploratory Standardization Check (age, fare)\n")
log(
    "This is an EDA-stage sanity check only. "
    "The modeling pipeline performs its own scaling.\n"
)

# Standardize the age and fare columns using the z-score method.
scaler = StandardScaler()
scaled = scaler.fit_transform(
    df[["age", "fare"]]
)
scaled_df = pd.DataFrame(
    scaled,
    columns=["age_z", "fare_z"]
)

# Compare mean and standard deviation before and after standardization.
before_after = pd.DataFrame(
    {
        "age (before)": [
            df["age"].mean(),
            df["age"].std()
        ],

        "age_z (after)": [
            scaled_df["age_z"].mean(),
            scaled_df["age_z"].std()
        ],

        "fare (before)": [
            df["fare"].mean(),
            df["fare"].std()
        ],

        "fare_z (after)": [
            scaled_df["fare_z"].mean(),
            scaled_df["fare_z"].std()
        ]
    },
    index=["mean", "std"]
)

log(
    before_after
    .round(3)
    .to_markdown()
)
log(
    "\nAfter z-score standardization, both columns "
    "have a mean close to 0 and a standard deviation close to 1."
)

# Save the cleaned EDA dataset.
df.to_csv(
    "titanic_cleaned_eda.csv",
    index=False
)
log(
    "\n---\n*Cleaned data saved to "
    "`titanic_cleaned_eda.csv`.*"
)

# Save all observations to the Markdown report.
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))
print(
    "\n\nDone. See eda_report.md and charts/ for full output."
)