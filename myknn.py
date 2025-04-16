import numpy as np
from collections import Counter

class KNN:
    def __init__(self, n_neighbors=5, p=2):
        self.k = n_neighbors
        self.p = p  # 1 for Manhattan, 2 for Euclidean

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _distance(self, x1, x2):
        if self.p == 1:  # Manhattan (L1)
            return np.sum(np.abs(x1 - x2))
        elif self.p == 2:  # Euclidean (L2)
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:
            raise ValueError("Only p=1 (L1) and p=2 (L2) are supported.")

    def _predict_single(self, x):
        # Compute distances from x to all training samples
        distances = [self._distance(x, x_train) for x_train in self.X_train]

        # Get indices of k smallest distances
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Return the most common label
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])
