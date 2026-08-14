# Titanic EDA Report (auto-generated from live computation)

## Task 1 -- Load & Profile

- Raw dataset loaded via `sns.load_dataset('titanic')` and saved to `titanic.csv` immediately after loading (one-time network/cache hit).
- Shape: **891 rows x 15 columns**

### df.info()
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
```
### df.describe()
|        |   survived |     pclass | sex   |      age |      sibsp |      parch |     fare | embarked   | class   | who   |   adult_male | deck   | embark_town   | alive   |   alone |
|:-------|-----------:|-----------:|:------|---------:|-----------:|-----------:|---------:|:-----------|:--------|:------|-------------:|:-------|:--------------|:--------|--------:|
| count  | 891        | 891        | 891   | 714      | 891        | 891        | 891      | 889        | 891     | 891   |          891 | 203    | 889           | 891     |     891 |
| unique | nan        | nan        | 2     | nan      | nan        | nan        | nan      | 3          | 3       | 3     |            2 | 7      | 3             | 2       |       2 |
| top    | nan        | nan        | male  | nan      | nan        | nan        | nan      | S          | Third   | man   |            1 | C      | Southampton   | no      |       1 |
| freq   | nan        | nan        | 577   | nan      | nan        | nan        | nan      | 644        | 491     | 537   |          537 | 59     | 644           | 549     |     537 |
| mean   |   0.383838 |   2.30864  | nan   |  29.6991 |   0.523008 |   0.381594 |  32.2042 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| std    |   0.486592 |   0.836071 | nan   |  14.5265 |   1.10274  |   0.806057 |  49.6934 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| min    |   0        |   1        | nan   |   0.42   |   0        |   0        |   0      | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| 25%    |   0        |   2        | nan   |  20.125  |   0        |   0        |   7.9104 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| 50%    |   0        |   3        | nan   |  28      |   0        |   0        |  14.4542 | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| 75%    |   1        |   3        | nan   |  38      |   1        |   0        |  31      | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |
| max    |   1        |   3        | nan   |  80      |   8        |   6        | 512.329  | nan        | nan     | nan   |          nan | nan    | nan           | nan     |     nan |

### Missing values (columns with any missing data)
- `deck`: 77.22% missing (688 rows)
- `age`: 19.87% missing (177 rows)
- `embarked`: 0.22% missing (2 rows)
- `embark_town`: 0.22% missing (2 rows)

## Task 2 -- Missing Value Handling

- `deck`: 77.22% missing -> column dropped.
- `age`: 19.87% missing -> filled with median (28.00).
- `embarked`: 0.22% missing -> 2 rows removed.
- `embark_town`: 0.22% missing -> 0 rows removed.

Shape after cleaning: **889 rows x 14 columns**

## Task 3 -- Univariate Analysis: age & fare

- `age`: IQR outlier bounds = [2.50, 54.50] -> **65 outliers** (7.3% of rows). Chart: `charts/univariate_age.png`
- `fare`: IQR outlier bounds = [-26.76, 65.66] -> **114 outliers** (12.8% of rows). Chart: `charts/univariate_fare.png`

- `fare`: mean = 32.10, median = 14.45, mode = 8.05 -> distribution is **right-skewed (mean > median > mode)**.

## Task 4 -- Bivariate Analysis

### Survival rate by sex (boolean masking)
- female: 0.740 (74.0%)
- male: 0.189 (18.9%)

### Survival rate by pclass (boolean masking)
- pclass 1: 0.626 (62.6%)
- pclass 2: 0.473 (47.3%)
- pclass 3: 0.242 (24.2%)

### Survival rate by sex AND pclass (boolean masking with &)
- female, pclass 1: 0.967 (96.7%)
- female, pclass 2: 0.921 (92.1%)
- female, pclass 3: 0.500 (50.0%)
- male, pclass 1: 0.369 (36.9%)
- male, pclass 2: 0.157 (15.7%)
- male, pclass 3: 0.135 (13.5%)

### Correlation heatmap: `charts/correlation_heatmap.png`

**Two strongest correlations (by absolute value):**
- `pclass` vs `fare`: r = -0.548 (negative correlation)
- `sibsp` vs `parch`: r = 0.415 (positive correlation)

Interpretation: the strongest relationship is between `pclass` and `fare` (r=-0.548), followed by `sibsp` and `parch` (r=0.415).

## Task 5 -- Multivariate Data Story (4+ charts)

**Chart 1 -- Survival by class & sex** (`charts/story_1_survival_by_class_sex.png`): female passengers in pclass 1 had the highest survival rate (96.7%), while male passengers in pclass 3 had the lowest (13.5%).

**Chart 2 -- Age by survival** (`charts/story_2_age_by_survival.png`): median age of survivors is 28.0 vs 28.0 for non-survivors.

**Chart 3 -- Fare vs age by survival** (`charts/story_3_fare_age_survival.png`).

**Chart 4 -- Survival by embarkation port** (`charts/story_4_survival_by_embarked.png`): 'C' had the highest survival rate.

**Chart 5 -- Pair plot** (`charts/story_5_pairplot.png`).

## Task 6 -- Exploratory Standardization Check (age, fare)

This is an EDA-stage sanity check only. The modeling pipeline performs its own scaling.

|      |   age (before) |   age_z (after) |   fare (before) |   fare_z (after) |
|:-----|---------------:|----------------:|----------------:|-----------------:|
| mean |         29.315 |           0     |          32.097 |            0     |
| std  |         12.985 |           1.001 |          49.698 |            1.001 |

After z-score standardization, both columns have a mean close to 0 and a standard deviation close to 1.

---
*Cleaned data saved to `titanic_cleaned_eda.csv`.*