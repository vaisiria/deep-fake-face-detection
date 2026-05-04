import os
import cv2
import numpy as np
import joblib

from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Paths
fake_path = "dataset/fake"
real_path = "dataset/real"

data = []
labels = []

# Preprocess function
def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None

    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray

# Load data
def load(folder, label):
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        img = preprocess(path)

        if img is not None:
            data.append(img)
            labels.append(label)

print("Loading dataset...")
load(fake_path, 0)
load(real_path, 1)

print("Total images:", len(data))

# Feature extraction
features = []

for img in data:
    lbp = local_binary_pattern(img, P=8, R=1)
    features.append(lbp.flatten())

X = np.array(features)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save models
joblib.dump(lr, "lr_model.pkl")
joblib.dump(knn, "knn_model.pkl")

print("Models saved successfully")

# Predictions
lr_pred = lr.predict(X_test)
knn_pred = knn.predict(X_test)

# Accuracy
lr_acc = accuracy_score(y_test, lr_pred)
knn_acc = accuracy_score(y_test, knn_pred)

print("LR Accuracy:", lr_acc)
print("KNN Accuracy:", knn_acc)

# Save accuracies
joblib.dump(lr_acc, "lr_acc.pkl")
joblib.dump(knn_acc, "knn_acc.pkl")

print("Saved Accuracies")