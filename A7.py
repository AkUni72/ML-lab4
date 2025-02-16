import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Select two features (e.g., O+ and A+ blood type percentages)
    X = df[['O+', 'A+']].values
    # Assign two arbitrary classes based on some condition (e.g., region-based classification)
    y = np.array([0 if i < len(df) // 2 else 1 for i in range(len(df))])
    return X, y

def train_knn(X, y, k):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    return knn, accuracy, X_test, y_test

def plot_decision_boundary(knn, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.title(title)
    plt.xlabel('O+ Blood Type %')
    plt.ylabel('A+ Blood Type %')
    plt.show()

def find_best_k(X, y):
    param_grid = {'n_neighbors': list(range(1, 20, 2))}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X, y)
    return grid_search.best_params_['n_neighbors']

# Load blood type data
X, y = load_data('bloodtypes.csv')

# Find best k
best_k = find_best_k(X, y)
print(f'Best k value found: {best_k}')

# Train and visualize with best k
knn, accuracy, X_test, y_test = train_knn(X, y, best_k)
print(f'Accuracy for best k={best_k}: {accuracy:.2f}')
plot_decision_boundary(knn, X, y, f'Optimal kNN Decision Boundary (k={best_k})')
