import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path, sheet_name):
    data = pd.ExcelFile(file_path)
    return data.parse(sheet_name)

def prepare_features_target(data, feature_cols, target_col):
    A = data[feature_cols].dropna()
    C = data[[target_col]].dropna()
    A, C = A.align(C, join='inner', axis=0)
    return A, C

def train_model(A, C):
    X = np.dot(np.linalg.pinv(A), C)
    return X

def predict(A, X):
    return np.dot(A, X)

def calculate_metrics(C, C_pred):
    mse = mean_squared_error(C, C_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((C - C_pred) / C)) * 100
    r2 = r2_score(C, C_pred)
    return mse, rmse, mape, r2

if __name__ == "__main__":
    file_path = "Lab Session Data.xlsx"
    sheet_name = "Purchase data"
    feature_cols = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
    target_col = 'Payment (Rs)'

    data = load_data(file_path, sheet_name)
    A, C = prepare_features_target(data, feature_cols, target_col)

    X = train_model(A, C)
    C_pred = predict(A, X)
    mse, rmse, mape, r2 = calculate_metrics(C, C_pred)

    print(f"MSE: {mse}\nRMSE: {rmse}\nMAPE: {mape}\nR² Score: {r2}")
