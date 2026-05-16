import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n── {model_name} ──")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k}: {v}")

    return metrics

def evaluate_all(trained_models, X_test, y_test):
    results = []
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(metrics)
    df = pd.DataFrame(results).set_index("Model")
    return df

def get_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)

def get_roc_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return fpr, tpr

if __name__ == "__main__":
    from data_ingestion import fetch_data
    from preprocessing import preprocess
    from modeling import train_all

    df = fetch_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess(df)
    trained = train_all(X_train, y_train)
    results = evaluate_all(trained, X_test, y_test)
    print("\n── Model Karşılaştırması ──")
    print(results.to_string())
