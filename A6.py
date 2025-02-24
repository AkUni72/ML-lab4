import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data(file_path):
    """Loads dataset and selects features for classification."""
    df = pd.read_csv(file_path)  # Read CSV file
    
    # Select two features (e.g., O+ and A+ blood type percentages)
    X = df[['O+', 'A+']].values  
    
    # Assign two arbitrary classes (e.g., region-based classification)
    y = np.array([0 if i < len(df) // 2 else 1 for i in range(len(df))])  
    
    return X, y

def train_knn(X, y, k):
    """Splits data, trains k-NN classifier, and returns model and accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
    knn = KNeighborsClassifier(n_neighbors=k)  # Initialize k-NN model
    knn.fit(X_train, y_train)  # Train the model
    accuracy = knn.score(X_test, y_test)  # Compute accuracy
    
    return knn, accuracy, X_test, y_test

def plot_decision_boundary(knn, X, y, title):
    """Plots decision boundary and training points."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  
    
    # Generate grid points for decision boundary
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))  
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])  # Predict classes for grid points
    Z = Z.reshape(xx.shape)  
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3)  
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')  
    plt.title(title)
    plt.xlabel('O+ Blood Type %')
    plt.ylabel('A+ Blood Type %')
    plt.show()

# Load dataset
X, y = load_data('bloodtypes.csv')

# Evaluate k-NN for different k values
for k in [1, 3, 5, 7]:
    knn, accuracy, X_test, y_test = train_knn(X, y, k)
    print(f'Accuracy for k={k}: {accuracy:.2f}')
    plot_decision_boundary(knn, X, y, f'kNN Decision Boundary (k={k})')
