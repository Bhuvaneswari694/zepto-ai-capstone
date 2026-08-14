
# Zepto Data & AI Platform Capstone

## Project Overview

This repository contains three interconnected modules that demonstrate an end-to-end AI/ML engineering workflow:

- `/data_pipeline` - Data collection, cleaning, transformation, storage, and analysis.
- `/analytics` - Exploratory data analysis and machine learning using the Titanic dataset.
- `/support_assistant` - A GenAI-powered support assistant for answering policy-related questions.

-------------------------------------------------------------------------------------------------------------------

## Repository Structure

```text
zepto-ai-capstone/
│
├── README.md
├── data_pipeline/
├── analytics/
└── support_assistant/
```

----------------------------------------------------------------------------------------------------------------------

## Module 1: Data Pipeline

### Objective

Build a complete data engineering pipeline that:

- Scrapes product data.
- Cleans and transforms the dataset.
- Performs currency conversion.
- Stores data in a relational database.
- Queries data using SQL and pandas.

### How to Run

1. Open the `data_pipeline` folder.
2. Install the required dependencies.
3. Run the pipeline scripts in sequence.

---------------------------------------------------------------------------------------------------------------

## Module 2: Analytics

### Objective

Build an end-to-end analytics workflow using the Titanic dataset.

### Deliverables

- Exploratory Data Analysis (EDA)
- Data cleaning
- Feature engineering
- Machine learning models
- Model evaluation
- Saved pipeline
- Visualizations

### How to Run

1. Open the `analytics` folder.
2. Install the required dependencies.

```bash
pip install -r requirements.txt
```

3. Run:

```bash
python 01_eda.py
python 02_modeling.py
```

-----------------------------------------------------------------------------------------------------------------


P. Bhuvaneswari
