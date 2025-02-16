import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def generate_training_data(n_samples=20, feature_range=(1, 10)):
    X_train = np.random.uniform(feature_range[0], feature_range[1], (n_samples, 2))
    y_train = np.random.randint(0, 2, n_samples)  # Class 0 (Blue) or Class 1 (Red)
    return X_train, y_train

def plot_training_data(X_train, y_train):
    colors = np.array(["blue", "red"])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors[y_train], edgecolor='k')
    plt.title("Training Data")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

X_train, y_train = generate_training_data()
plot_training_data(X_train, y_train)