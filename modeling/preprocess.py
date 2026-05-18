import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
TARGET = "stroke"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=["id"])

    df = df[df["gender"] != "Other"].reset_index(drop=True)

    df["bmi_missing"] = df["bmi"].isnull().astype(int)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 60, 120],
        labels=["young", "middle", "senior"],
    )

    df["glucose_category"] = pd.cut(
        df["avg_glucose_level"],
        bins=[0, 100, 125, 500],
        labels=["normal", "prediabetic", "diabetic"],
    )

    return df


class AgeBasedBMIImputer(BaseEstimator, TransformerMixin):
    def __init__(self, age_col="age", bmi_col="bmi"):
        self.age_col = age_col
        self.bmi_col = bmi_col

    def fit(self, X, y=None):
        X = X.copy()
        age_bins = pd.cut(X[self.age_col], bins=[0, 40, 60, 120])
        self.group_medians_ = X.groupby(age_bins, observed=False)[self.bmi_col].median()
        self.global_median_ = X[self.bmi_col].median()
        return self

    def transform(self, X):
        X = X.copy()
        age_bins = pd.cut(X[self.age_col], bins=[0, 40, 60, 120])

        for group in self.group_medians_.index:
            mask = (age_bins == group) & (X[self.bmi_col].isnull())
            fill_value = self.group_medians_[group]
            if pd.isnull(fill_value):
                fill_value = self.global_median_
            X.loc[mask, self.bmi_col] = fill_value

        X[self.bmi_col] = X[self.bmi_col].fillna(self.global_median_)
        return X


def build_preprocessor() -> Pipeline:
    numeric_cols = ["age", "avg_glucose_level", "bmi"]
    binary_cols = ["hypertension", "heart_disease", "ever_married", "Residence_type", "bmi_missing"]
    ohe_cols = ["gender", "work_type", "smoking_status", "age_group", "glucose_category"]

    numeric_tf = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    binary_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    ohe_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_tf = ColumnTransformer(transformers=[
        ("num", numeric_tf, numeric_cols),
        ("bin", binary_tf, binary_cols),
        ("ohe", ohe_tf, ohe_cols),
    ])

    preprocessor = Pipeline(steps=[
        ("bmi_imputer", AgeBasedBMIImputer()),
        ("column_tf", column_tf),
    ])

    return preprocessor


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fetch_data import fetch_data

    df = fetch_data()
    print("Ham veri:", df.shape)

    df_clean = clean_data(df)
    print("Temizlenmis veri:", df_clean.shape)
    print("\nYeni sutunlar:")
    print(df_clean[["age_group", "glucose_category", "bmi_missing"]].head())

    preprocessor = build_preprocessor()
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]

    X_transformed = preprocessor.fit_transform(X, y)
    print("\nPreprocessing sonrasi sekil:", X_transformed.shape)