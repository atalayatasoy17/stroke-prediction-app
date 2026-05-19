import json
import os
import sys

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

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


def optimize_xgb(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        model = TunedThresholdClassifierCV(
            estimator=XGBClassifier(
                **params, random_state=SEED, eval_metric="logloss"
            ),
            scoring="f1",
            cv=cv,
            random_state=SEED,
        )

        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring="f1")
        return scores["test_score"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print(f"\nEn iyi XGBoost parametreleri: {study.best_params}")
    print(f"En iyi F1: {study.best_value:.3f}")

    return study.best_params


def build_models(xgb_params=None):
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

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
        }

    xgb = TunedThresholdClassifierCV(
        estimator=XGBClassifier(
            **xgb_params, random_state=SEED, eval_metric="logloss"
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

    print("XGBoost hiperparametre optimizasyonu yapiliyor...")
    xgb_params = optimize_xgb(X_train, y_train)
    models = build_models(xgb_params=xgb_params)

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
        ("model", build_models(xgb_params=xgb_params)[best_name]),
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

    print("\n--- Threshold Analysis ---")
    print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print("-" * 42)
    for thresh in np.linspace(0.05, 0.35, num=13):
        y_pred_thresh = (y_prob >= thresh).astype(int)
        r = recall_score(y_test, y_pred_thresh)
        p = precision_score(y_test, y_pred_thresh, zero_division=0)
        f = f1_score(y_test, y_pred_thresh)
        print(f"{thresh:>10.3f} {r:>8.3f} {p:>10.3f} {f:>8.3f}")

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
    print("Sonuclar kaydedildi: modeling/artifacts/results.json")

    print("\n--- Feature Importance (RandomForest) ---")
    rf_model = best_pipe.named_steps["model"]
    if hasattr(rf_model, "estimator_"):
        rf_estimator = rf_model.estimator_
    else:
        rf_estimator = rf_model

    feature_names = best_pipe.named_steps["preprocessor"].named_steps["column_tf"].get_feature_names_out()
    importances = rf_estimator.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print(feat_imp.head(10))

    feat_imp.head(10).to_json("modeling/artifacts/feature_importance.json")
    print("Feature importance kaydedildi.")

    return results, best_name


if __name__ == "__main__":
    evaluate_and_train()