import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fetch_data import fetch_data
from modeling.preprocess import TARGET, SEED, build_preprocessor, clean_data


def prepare_data():
    df = fetch_data()
    df = clean_data(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_models():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    baseline = DummyClassifier(strategy="most_frequent")

    logreg = TunedThresholdClassifierCV(
        estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        scoring="f1",
        cv=cv,
        random_state=SEED,
    )

    rf = TunedThresholdClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=200, max_depth=7, random_state=SEED
        ),
        scoring="f1",
        cv=cv,
        random_state=SEED,
    )

    xgb = TunedThresholdClassifierCV(
        estimator=XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=SEED,
            eval_metric="logloss",
        ),
        scoring="f1",
        cv=cv,
        random_state=SEED,
    )

    return {
        "Baseline": baseline,
        "LogisticRegression": logreg,
        "RandomForest": rf,
        "XGBoost": xgb,
    }


def evaluate_and_train():
    X_train, X_test, y_train, y_test = prepare_data()
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scorers = ["f1", "recall", "roc_auc"]
    results = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=scorers, return_estimator=True
        )

        results[name] = {
            "f1": cv_res["test_f1"].mean(),
            "recall": cv_res["test_recall"].mean(),
            "roc_auc": cv_res["test_roc_auc"].mean(),
        }

        print(f"\n{name}")
        print(f"  F1:      {results[name]['f1']:.3f}")
        print(f"  Recall:  {results[name]['recall']:.3f}")
        print(f"  ROC-AUC: {results[name]['roc_auc']:.3f}")

        if name != "Baseline":
            thresholds = []
            for estimator in cv_res["estimator"]:
                tuned_model = estimator.named_steps["model"]
                thresholds.append(tuned_model.best_threshold_)
            avg_threshold = np.mean(thresholds)
            results[name]["threshold"] = avg_threshold
            print(f"  Threshold: {avg_threshold:.3f}")

    best_name = max(
        [k for k in results if k != "Baseline"],
        key=lambda k: results[k]["f1"],
    )
    print(f"\nEn iyi model: {best_name}")

    best_pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", build_models()[best_name]),
    ])
    best_pipe.fit(X_train, y_train)

    os.makedirs("modeling/artifacts", exist_ok=True)
    joblib.dump(best_pipe, "modeling/artifacts/best_model.pkl")
    print("Model kaydedildi: modeling/artifacts/best_model.pkl")

    print(f"\n--- Final Test Evaluation ({best_name}) ---")
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    test_roc = roc_auc_score(y_test, y_prob)
    print(f"Test ROC-AUC: {test_roc:.3f}")

    results["test_evaluation"] = {
        "model": best_name,
        "roc_auc": round(test_roc, 3),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open("modeling/artifacts/results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)
    return results, best_name


if __name__ == "__main__":
    evaluate_and_train()