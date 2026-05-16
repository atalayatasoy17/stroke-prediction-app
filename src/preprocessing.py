import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def clean_data(df):
    df["bmi"] = df["bmi"].fillna(
        df.groupby("gender")["bmi"].transform("mean")
    )
    df = df.drop(columns=["id"], errors="ignore")
    return df

def split_data(df):
    df = clean_data(df)
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=17, stratify=y, shuffle=True
    )
    return X_train, X_test, y_train, y_test

def preprocess(X_train, X_test, y_train, model_type="logistic"):
    cat_x = X_train.select_dtypes(["object"])
    numeric_x = X_train.select_dtypes([int, float])

    if model_type == "logistic":
        encoder = OneHotEncoder(drop="first", sparse_output=False)
    else:
        encoder = OneHotEncoder(drop=None, sparse_output=False)

    scaler = StandardScaler()

    encoded_train = encoder.fit_transform(cat_x)
    scaled_train = scaler.fit_transform(numeric_x)
    X_train_transformed = np.concatenate([scaled_train, encoded_train], axis=1)

    cat_test = X_test.select_dtypes(["object"])
    numeric_test = X_test.select_dtypes([int, float])
    encoded_test = encoder.transform(cat_test)
    scaled_test = scaler.transform(numeric_test)
    X_test_transformed = np.concatenate([scaled_test, encoded_test], axis=1)

    return X_train_transformed, X_test_transformed, encoder, scaler

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=17)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

if __name__ == "__main__":
    from data_ingestion import fetch_data
    df = fetch_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_t, X_test_t, encoder, scaler = preprocess(X_train, X_test, y_train)
    X_train_t, y_train = apply_smote(X_train_t, y_train)
    print("Preprocessing tamamlandi!")
    print(f"X_train: {X_train_t.shape}")
    print(f"X_test: {X_test_t.shape}")