import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def load_data(file_name):
    """Loads dataset and handles missing values."""
    df = pd.read_csv(file_name)  # Read CSV file into a DataFrame
    return df.dropna()  # Remove any rows with missing values

def preprocess_data(df):
    """Creates target variable based on O+ blood type threshold."""
    # Create a binary target column: 1 if O+ is greater than 40, else 0
    df['High_O+'] = df['O+'].apply(lambda x: 1 if x > 40 else 0)

    # Define feature columns (other blood type distributions)
    features = ['A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
    
    # Define target column
    target = 'High_O+'
    
    return df[features], df[target]  # Return features and target as separate DataFrames

def train_knn(X_train, y_train, k=3):
    """Trains a k-NN classifier."""
    knn = KNeighborsClassifier(n_neighbors=k)  # Initialize k-NN model
    knn.fit(X_train, y_train)  # Train the model
    return knn  # Return trained model

def evaluate_model(y_test, y_pred):
    """Evaluates the model using precision, recall, and F1-score."""
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Compute classification metrics
    precision = precision_score(y_test, y_pred)  # Precision score
    recall = recall_score(y_test, y_pred)  # Recall score
    f1 = f1_score(y_test, y_pred)  # F1-score

    # Print results
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

def main():
    """Main function to execute the program."""
    df = load_data("bloodtypes.csv")  # Load dataset
    X, y = preprocess_data(df)  # Preprocess data (feature selection & target creation)
    
    # Split data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train k-NN model
    model = train_knn(X_train, y_train, k=3)
    
    # Predict target values for test data
    y_pred = model.predict(X_test)

    # Evaluate model performance
    evaluate_model(y_test, y_pred)

# Run the program only if executed directly
if __name__ == "__main__":
    main()
