import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def generate_training_data(n_samples=20, feature_range=(1, 10)):
    """Generates random training data with two features."""
    X_train = np.random.uniform(feature_range[0], feature_range[1], (n_samples, 2))  # Generate feature values
    y_train = np.random.randint(0, 2, n_samples)  # Generate class labels (0 or 1)
    return X_train, y_train

def plot_training_data(X_train, y_train):
    """Plots training data with different colors for each class."""
    colors = np.array(["blue", "red"])  # Define colors for classes
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors[y_train], edgecolor='k')  # Scatter plot with colors
    plt.title("Training Data")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

# Generate and plot training data
X_train, y_train = generate_training_data()
plot_training_data(X_train, y_train)
