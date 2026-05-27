# Predicting Final Calculus Grades Using ALEKS Assessment Metrics

This repository contains an end-to-end Machine Learning and Data Science project designed to predict students' final numerical Calculus grades. Utilizing performance data derived from the ALEKS mastery platform, this project implements an analytical framework consisting of thorough Exploratory Data Analysis (EDA), data preprocessing, and a Random Forest classification pipeline to identify key determinants of student academic success.

## Project Overview
Accurate prediction of student performance is vital in educational analytics, enabling educators to provide timely interventions and optimize resource allocation. This project systematically evaluates 1,988 student records across 20 distinct metrics—ranging from initial assessment benchmarks to final mastery metrics—to forecast 10 discrete final grade categories.

## Key Findings & Insights
The Exploratory Data Analysis (EDA) established a highly definitive conclusion regarding student outcomes:
* **Mastery Over Readiness:** Final mastery metrics (such as the *Post-Test Score* and *Weighted Latest Grade*) exhibit a significantly stronger correlation with top final grades (e.g., 3.0 and 5.0) than initial readiness scores or net learning gains.
* **The Learning Plateau:** Net learning gain alone was not the primary driver for achieving maximum performance; sustained mastery at the end of the course serves as the most accurate predictor of success.

## Data & Machine Learning Pipeline

### 1. Data Cleaning & Engineering
- Checked and resolved structural anomalies, null records, and type mappings.
- Separated student performance features ($X$) from the multi-class target labels ($y$, representing the final numerical grades).
- Utilized a `LabelEncoder` to cleanly transform categorical or discrete target classes into structured numeric indexes.

### 2. Train-Test Split
- Partitioned the 1,988 records into an **70% Training set** (to construct and train the estimator) and a **30% Testing set** (586 records held out for final model evaluation).

### 3. Predictive Modeling
- Implemented a **Random Forest Classifier** from `scikit-learn`. Random Forests are uniquely suited for educational data tables as they naturally capture non-linear interactions between disparate metrics without requiring complex scaling.

### 4. Evaluation Performance
The model was verified against the unseen 30% test block, resulting in the following outputs:
* **Overall Accuracy Score:** `52.05%` (0.5205)
* **Strengths:** Precision, Recall, and F1-Scores confirmed highly effective predictions among the dominant grade classes (`5.0` and `3.0`).
* **Challenges:** Classification boundaries proved more complex when distinguishing between micro-adjacent lower-range grades (e.g., `1.0` vs. `1.25`), where the variance in foundational ALEKS scores was mathematically narrow.

## Tech Stack & Libraries
- **Language:** Python
- **Data Engineering & EDA:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning & Preprocessing:** `scikit-learn` (`RandomForestClassifier`, `LabelEncoder`, `accuracy_score`, `classification_report`, `confusion_matrix`)
- **Development Interface:** Jupyter Notebooks
