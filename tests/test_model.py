import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_data import fetch_data
from modeling.preprocess import clean_data, TARGET, SEED
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
import joblib


@pytest.fixture(scope="module")
def model():
    return joblib.load("modeling/artifacts/best_model.pkl")


@pytest.fixture(scope="module")
def test_split():
    df = fetch_data()
    df = clean_data(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_test, y_test


@pytest.fixture(scope="module")
def sample_data():
    df = fetch_data()
    df = clean_data(df)
    return df.drop(columns=[TARGET])


def test_model_loads(model):
    """Best model loads successfully."""
    assert model is not None


def test_model_predicts(model, test_split):
    """Model produces binary predictions without errors."""
    X_test, _ = test_split
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})


def test_model_predicts_probabilities(model, test_split):
    """Model produces valid probability scores between 0 and 1."""
    X_test, _ = test_split
    probabilities = model.predict_proba(X_test)[:, 1]
    assert len(probabilities) == len(X_test)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


def test_model_roc_auc_above_threshold(model, test_split):
    """Model ROC-AUC exceeds minimum acceptable threshold (0.75)."""
    X_test, y_test = test_split
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    assert auc > 0.75, f"ROC-AUC {auc:.3f} is below acceptable threshold 0.75"


def test_model_recall_above_threshold(model, test_split):
    """Model Recall exceeds minimum acceptable threshold (0.40)."""
    X_test, y_test = test_split
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    assert recall > 0.40, f"Recall {recall:.3f} is below acceptable threshold 0.40"


def test_high_risk_patient_scores_higher(model, sample_data):
    """High-risk patient profile scores higher than low-risk profile."""
    high_risk = sample_data.iloc[0:1].copy()
    high_risk["age"] = 80
    high_risk["avg_glucose_level"] = 250.0
    high_risk["hypertension"] = 1
    high_risk["heart_disease"] = 1

    low_risk = sample_data.iloc[0:1].copy()
    low_risk["age"] = 25
    low_risk["avg_glucose_level"] = 80.0
    low_risk["hypertension"] = 0
    low_risk["heart_disease"] = 0

    high_score = model.predict_proba(high_risk)[:, 1][0]
    low_score = model.predict_proba(low_risk)[:, 1][0]

    assert high_score > low_score, (
        f"High-risk score ({high_score:.3f}) should exceed "
        f"low-risk score ({low_score:.3f})"
    )


def test_model_not_biased_toward_one_class(model, test_split):
    """Model predicts both classes — not stuck predicting only one class."""
    X_test, _ = test_split
    predictions = model.predict(X_test)
    assert 0 in predictions, "Model never predicts no-stroke"
    assert 1 in predictions, "Model never predicts stroke"

