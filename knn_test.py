import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN  # Assuming knn.py has the KNN class implementation

# Color map for visualization
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

# Load iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Uncomment to visualize data
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

# Initialize the KNN classifier
clf = KNN(k=5)
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Calculate accuracy
acc = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {acc}")
