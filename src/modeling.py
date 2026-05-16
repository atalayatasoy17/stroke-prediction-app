import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def train_models(X_train, y_train):
    models = {
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
            max_iter=200,
            early_stopping=True,
            random_state=17
        )
    }
    
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    
    return trained