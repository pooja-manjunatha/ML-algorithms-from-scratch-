import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):  # Global function for Euclidean distance
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):  # Corrected the constructor
        self.k = k

    # Fit training samples and training labels
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Predict new samples
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]  # Call _predict for each sample
        return np.array(predicted_labels)

    # Helper method for predicting one sample
    def _predict(self, x):
        # Compute distances between x and all training samples
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
