import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_data import fetch_data
from modeling.preprocess import clean_data, build_preprocessor, TARGET, SEED
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def raw_data():
    return fetch_data()


@pytest.fixture(scope="module")
def clean_df(raw_data):
    return clean_data(raw_data)


def test_data_loaded(raw_data):
    """Dataset loads successfully with expected shape."""
    assert raw_data is not None
    assert len(raw_data) > 5000
    assert "stroke" in raw_data.columns


def test_clean_data_removes_gender_other(raw_data):
    """gender=Other row is removed during cleaning."""
    cleaned = clean_data(raw_data)
    assert "Other" not in cleaned["gender"].values


def test_clean_data_creates_features(clean_df):
    """Feature engineering creates expected columns."""
    assert "age_group" in clean_df.columns
    assert "glucose_category" in clean_df.columns
    assert "bmi_missing" in clean_df.columns


def test_preprocessor_output_shape(clean_df):
    """Preprocessor transforms data to expected feature count."""
    X = clean_df.drop(columns=[TARGET])
    y = clean_df[TARGET]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)
    assert X_transformed.shape[0] == len(X_train)
    assert X_transformed.shape[1] > 0


def test_no_missing_values_after_preprocessing(clean_df):
    """No missing values remain after preprocessing."""
    X = clean_df.drop(columns=[TARGET])
    y = clean_df[TARGET]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)
    assert not np.isnan(X_transformed).any()


def test_stratified_split_preserves_ratio(clean_df):
    """Train/test split preserves class ratio."""
    X = clean_df.drop(columns=[TARGET])
    y = clean_df[TARGET]
    _, _, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.01