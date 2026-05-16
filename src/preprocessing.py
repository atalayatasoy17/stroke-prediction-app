import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess(df):
    df["bmi"] = df["bmi"].fillna(df.groupby("gender")["bmi"].transform("mean"))
    df = df.drop(columns=["id"])
    
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=17, stratify=y, shuffle=True
    )
    
    cat_x = X_train.select_dtypes(["object"])
    numeric_X = X_train.select_dtypes([int, float])
    
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    scaler = StandardScaler()
    
    encoded_train = encoder.fit_transform(cat_x)
    scaled_train = scaler.fit_transform(numeric_X)
    X_train_transformed = np.concatenate([scaled_train, encoded_train], axis=1)
    
    cat_test = X_test.select_dtypes(["object"])
    num_test = X_test.select_dtypes([int, float])
    encoded_test = encoder.transform(cat_test)
    scaled_test = scaler.transform(num_test)
    X_test_transformed = np.concatenate([scaled_test, encoded_test], axis=1)
    
    smote = SMOTE(random_state=17)
    X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)
    
    return X_resampled, X_test_transformed, y_resampled, y_test, encoder, scaler