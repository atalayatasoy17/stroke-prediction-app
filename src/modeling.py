import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")

def get_models():
    return {
        "Logistic Regression": LogisticRegressionCV(
            Cs=10,
            penalty="l1",
            solver="liblinear",
            cv=5,
            scoring="f1",
            class_weight="balanced",
            random_state=17
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=17
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=17
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,
            random_state=17
        ),
    }

def train_all(X_train, y_train):
    models = get_models()
    trained = {}
    for name, model in models.items():
        print(f"{name} egitiliyor...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"{name} tamamlandi.")
    return trained

def save_models(trained_models):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in trained_models.items():
        path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, path)
        print(f"Kaydedildi: {path}")

def load_models():
    models = {}
    if not os.path.exists(MODEL_DIR):
        return models
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".pkl"):
            path = os.path.join(MODEL_DIR, fname)
            name = fname.replace("_", " ").replace(".pkl", "").title()
            models[name] = joblib.load(path)
    return models

if __name__ == "__main__":
    from data_ingestion import fetch_data
    from preprocessing import clean_data, split_data, preprocess, apply_smote
    
    df = fetch_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_t, X_test_t, encoder, scaler = preprocess(X_train, X_test, y_train)
    X_train_t, y_train = apply_smote(X_train_t, y_train)
    trained = train_all(X_train_t, y_train)
    save_models(trained)
    print("Tum modeller egitildi ve kaydedildi!")