import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path, sheet_name):
    """Loads data from an Excel file."""
    data = pd.ExcelFile(file_path)  # Read the Excel file
    return data.parse(sheet_name)  # Parse the specified sheet

def prepare_features_target(data, feature_cols, target_col):
    """Prepares feature matrix A and target matrix C."""
    A = data[feature_cols].dropna()  # Drop missing values in feature columns
    C = data[[target_col]].dropna()  # Drop missing values in target column

    # Align feature and target matrices to ensure matching rows
    A, C = A.align(C, join='inner', axis=0)
    
    return A, C

def train_model(A, C):
    """Trains a linear regression model using the pseudo-inverse method."""
    X = np.dot(np.linalg.pinv(A), C)  # Compute model coefficients
    return X

def predict(A, X):
    """Generates predictions using the trained model."""
    return np.dot(A, X)  # Compute predicted target values

def calculate_metrics(C, C_pred):
    """Computes evaluation metrics for regression performance."""
    mse = mean_squared_error(C, C_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mape = np.mean(np.abs((C - C_pred) / C)) * 100  # Mean Absolute Percentage Error
    r2 = r2_score(C, C_pred)  # R² Score (coefficient of determination)
    
    return mse, rmse, mape, r2

if __name__ == "__main__":
    file_path = "Lab Session Data.xlsx"  # Define file path
    sheet_name = "Purchase data"  # Specify sheet name
    
    # Define feature columns and target variable
    feature_cols = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
    target_col = 'Payment (Rs)'

    data = load_data(file_path, sheet_name)  # Load dataset
    A, C = prepare_features_target(data, feature_cols, target_col)  # Prepare features and target

    X = train_model(A, C)  # Train model
    C_pred = predict(A, X)  # Make predictions
    mse, rmse, mape, r2 = calculate_metrics(C, C_pred)  # Compute evaluation metrics

    # Print evaluation results
    print(f"MSE: {mse}\nRMSE: {rmse}\nMAPE: {mape}\nR² Score: {r2}")
