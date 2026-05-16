import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
TARGET = "stroke"

def clean_data(df):
    df = df.copy()
    df.drop(columns=["id"], inplace=True)
    df = df[df["gender"] != "Other"]
    df["bmi"] = df.groupby("gender")["bmi"].transform(
        lambda x: x.fillna(x.mean())
    )
    print(f"Temizlendi. Satır: {df.shape[0]}")
    return df

def preprocess(df, apply_smote=True):
    df = clean_data(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"SMOTE sonrası train: {X_train.shape}")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    from data_ingestion import fetch_data
    df = fetch_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess(df)
    print("Preprocessing tamamlandı!")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
