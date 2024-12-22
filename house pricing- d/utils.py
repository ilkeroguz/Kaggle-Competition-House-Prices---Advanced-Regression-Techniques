import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def preprocess_data(df, feature_columns=None):
    if feature_columns is None:
        df = pd.get_dummies(df, drop_first=True)
        return df, df.columns.tolist()
    else:
        df = pd.get_dummies(df, drop_first=True)
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        return df[feature_columns], feature_columns

def encode_categorical_columns(df):
    encoders = {}
    for column in df.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        df[column + "_encoded"] = encoder.fit_transform(df[column])
        encoders[column] = encoder
    return df, encoders

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)