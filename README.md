# Stroke Prediction 

An end-to-end machine learning application for predicting stroke risk from patient health data. Built as the final project for DS 570.

## Overview

Stroke is one of the leading causes of death and disability worldwide. Early identification of high-risk patients enables preventive interventions. This project builds a complete ML pipeline that:

- Explores patient health data to uncover stroke risk patterns
- Handles real-world data challenges (missing values, class imbalance, outliers)
- Trains and evaluates multiple ML models with proper cross-validation
- Provides an interactive dashboard for risk assessment

The project emphasizes **principled data science** over raw model performance — every preprocessing and modeling decision is motivated by exploratory analysis and validated by results.

## Dataset

- **Source:** [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Author:** fedesoriano
- **License:** CC BY-SA 4.0
- **Size:** 5,110 patients, 11 features
- **Target:** Binary stroke outcome (4.9% positive rate)

Features include demographic information (age, gender, marital status), medical history (hypertension, heart disease), lifestyle factors (smoking status, work type), and clinical measurements (BMI, average glucose level).

## Key Findings

- **Severe class imbalance** (95/5) — accuracy is misleading; we use F1, Recall, and ROC-AUC
- **BMI missingness is informative** — patients with missing BMI have 5x higher stroke rate (Missing Not At Random)
- **Compounding effect** — patients with both hypertension and heart disease show 4x higher stroke rate than those with neither
- **Glucose threshold effect** — risk jumps only after crossing the diabetic threshold (>125 mg/dL)
- **Smoking paradox** — formerly smoked has the highest stroke rate, suggesting reverse causality

## Project Structure

```
stroke-prediction-app/
├── app/
│   └── dashboard.py            # Streamlit dashboard (3 pages)
├── data/
│   └── fetch_data.py           # Kaggle API data fetcher
├── modeling/
│   ├── preprocess.py           # Custom transformers + pipeline
│   ├── train.py                # Training script with Optuna HPO
│   └── artifacts/              # Saved models and results (gitignored)
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
├── tests/
│   └── test_pipeline.py        # Pipeline tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## Features

### Dashboard (3 Pages)

**1. Exploratory Data Analysis**
- Dataset overview and class distribution
- Univariate and bivariate analysis
- Domain-grounded interpretations (MNAR, confounding, reverse causality)
- Each visualization paired with analytical commentary

**2. Model Results**
- Preprocessing pipeline visualization
- Model selection justification (chosen vs rejected)
- Cross-validation strategy with fold-level results
- Best model evaluation (Confusion Matrix, ROC Curve, PR Curve)
- Interactive threshold analysis
- Feature importance
- Error analysis identifying model blind spots

**3. Risk Predictor**
- Interactive patient input form
- Real-time risk prediction with adjustable threshold
- Risk factor breakdown
- Clinical safety notes based on error analysis

### Modeling Pipeline

- **Custom transformer:** `AgeBasedBMIImputer` for principled BMI imputation
- **Feature engineering:** age groups, glucose categories, missingness flags
- **Models:** Logistic Regression, Random Forest, XGBoost + Dummy baseline
- **Threshold tuning:** `TunedThresholdClassifierCV` for imbalanced data
- **Hyperparameter optimization:** Optuna (30 trials) for XGBoost

## Results

| Model | F1 | Recall | ROC-AUC | Threshold |
|-------|-----|--------|---------|-----------|
| DummyClassifier (Baseline 1) | 0.000 | 0.000 | 0.500 | - |
| Logistic Regression (Baseline 2) | 0.258 | 0.503 | 0.853 | 0.115 |
| **Random Forest (Best)** | **0.293** | **0.493** | **0.848** | **0.131** |
| XGBoost (Optuna) | 0.281 | 0.453 | 0.841 | 0.129 |

**Final test set performance (Random Forest):**
- F1: 0.290 (matches CV — no overfitting)
- ROC-AUC: 0.821
- Recall: 0.54 (catches 27 of 50 stroke cases)
- Precision: 0.20

## Setup and Installation

### Prerequisites

- Python 3.11+
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Local Setup

```bash
git clone https://github.com/atalayatasoy17/stroke-prediction-app.git
cd stroke-prediction-app
pip install -r requirements.txt
```

### Train the Model

```bash
python modeling/train.py
```

This downloads the dataset, trains all models with 5-fold CV, runs Optuna optimization, and saves the best model to `modeling/artifacts/best_model.pkl`.

### Run the Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard opens at `http://localhost:8501`.

## Docker

The entire project runs in a single Docker container with one command.

### Build

```bash
docker build \
  --build-arg KAGGLE_USERNAME=your_username \
  --build-arg KAGGLE_KEY=your_api_key \
  -t stroke-prediction-app .
```

### Run

```bash
docker run -p 8501:8501 stroke-prediction-app
```

Then open `http://localhost:8501` in your browser.

The Docker build automatically fetches the dataset, trains the model, and starts the dashboard.

## Technologies

- **Data:** pandas, NumPy
- **Modeling:** scikit-learn, XGBoost, Optuna
- **Visualization:** Plotly
- **Dashboard:** Streamlit
- **Containerization:** Docker

## Limitations

- F1 score of 0.29 reflects the difficulty of predicting stroke from only 249 positive cases
- Model catches 54% of stroke cases at optimal threshold — 46% miss rate
- Error analysis identified a blind spot: younger patients (40-65) with normal glucose levels
- Data collection method and time period are not documented by the original author
- Results may not generalize beyond the population in the dataset

## Future Work

- Acquire larger and more diverse stroke datasets
- Incorporate additional clinical features (blood pressure readings, cholesterol)
- Experiment with class-weighted training and ensemble methods
- Investigate the Unknown smoking status group with additional features

## Author

Built by Atalay Atasoy for DS 570 — Final Project.

## License

This project is for educational purposes. The dataset is licensed under CC BY-SA 4.0.