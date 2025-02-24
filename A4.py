import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def generate_training_data(n_samples=20, feature_range=(1, 10)):
    """Generates random training data with two features and binary class labels."""
    X_train = np.random.uniform(feature_range[0], feature_range[1], (n_samples, 2))  # Feature values
    y_train = np.random.randint(0, 2, n_samples)  # Class labels (0: Blue, 1: Red)
    return X_train, y_train

def generate_test_data(x_range=(0, 10), y_range=(0, 10), step=0.1):
    """Generates a grid of test points for classification."""
    x_values = np.arange(x_range[0], x_range[1], step)  # X-axis values
    y_values = np.arange(y_range[0], y_range[1], step)  # Y-axis values
    X_test = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)  # Create a meshgrid
    return X_test

def classify_and_plot(X_train, y_train, X_test, k):
    """Trains k-NN and visualizes decision boundaries and training data."""
    knn = KNeighborsClassifier(n_neighbors=k)  # Initialize k-NN classifier
    knn.fit(X_train, y_train)  # Train the classifier
    y_pred = knn.predict(X_test)  # Predict class labels for test data
    
    colors = np.array(["blue", "red"])  # Define colors for classes
    
    # Plot classification regions
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors[y_pred], alpha=0.3, s=10)  
    # Plot training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors[y_train], edgecolor='k', marker='o', s=50)  
    
    plt.title(f"kNN Classification with k={k}")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

# Generate training and test data
X_train, y_train = generate_training_data()
X_test = generate_test_data()

# Train and visualize k-NN classification
classify_and_plot(X_train, y_train, X_test, k=3)
