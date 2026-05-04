
import streamlit as st
import cv2
import numpy as np
import joblib
import os
import random
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# -----------------------
# Load models and accuracy
# -----------------------
lr = joblib.load("lr_model.pkl")
knn = joblib.load("knn_model.pkl")

lr_acc = joblib.load("lr_acc.pkl")
knn_acc = joblib.load("knn_acc.pkl")

st.markdown("<h1 style='text-align: center;'>Deepfake Face Detection System</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -----------------------
# Model Accuracy
# -----------------------
st.subheader("Model Accuracy")

col1, col2 = st.columns(2)

with col1:
    st.metric("Logistic Regression Accuracy", f"{lr_acc:.2f}")

with col2:
    st.metric("KNN Accuracy", f"{knn_acc:.2f}")

# -----------------------
# Model Comparison Graph
# -----------------------
st.subheader("Model Comparison")

models = ["Logistic Regression", "KNN"]
accuracy = [lr_acc, knn_acc]

fig, ax = plt.subplots()
ax.bar(models, accuracy)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Performance Comparison")

for i, v in enumerate(accuracy):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center')

st.pyplot(fig)

# -----------------------
# Conclusion
# -----------------------
st.subheader("Conclusion")

if lr_acc > knn_acc:
    st.success("Logistic Regression performs better for this dataset")
else:
    st.success("KNN performs better for this dataset")

# -----------------------
# Preprocessing Function
# -----------------------
def preprocess(img):
    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1)
    return lbp.flatten().reshape(1, -1)

# -----------------------
# Upload Image
# -----------------------
st.subheader("Upload Image")

uploaded = st.file_uploader("Choose image", type=["jpg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image")

    features = preprocess(img)

    lr_pred = lr.predict(features)[0]
    knn_pred = knn.predict(features)[0]

    # Confidence
    lr_prob = lr.predict_proba(features)[0]
    knn_prob = knn.predict_proba(features)[0]

    lr_conf = max(lr_prob)
    knn_conf = max(knn_prob)

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Logistic Regression")
        st.write("Prediction:", "Real" if lr_pred else "Fake")
        st.progress(float(lr_conf))

    with col2:
        st.subheader("KNN")
        st.write("Prediction:", "Real" if knn_pred else "Fake")
        st.progress(float(knn_conf))
    # Final decision
    st.subheader("Final Decision")

    if lr_acc > knn_acc:
        st.success(f"Final Prediction (Best Model - LR): {'Real' if lr_pred else 'Fake'}")
    else:
        st.success(f"Final Prediction (Best Model - KNN): {'Real' if knn_pred else 'Fake'}")
st.markdown("---")
# -----------------------
# Random Image Demo
# -----------------------
st.subheader("Test Random Image")

if st.button("Show Random Image"):
    folder = random.choice(["dataset/fake", "dataset/real"])
    img_name = random.choice(os.listdir(folder))
    path = os.path.join(folder, img_name)

    img = cv2.imread(path)

    st.image(img, caption="Random Image")

    features = preprocess(img)

    lr_pred = lr.predict(features)[0]
    knn_pred = knn.predict(features)[0]

    # Confidence
    lr_prob = lr.predict_proba(features)[0]
    knn_prob = knn.predict_proba(features)[0]

    lr_conf = max(lr_prob)
    knn_conf = max(knn_prob)

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Logistic Regression")
        st.write("Prediction:", "Real" if lr_pred else "Fake")
        st.progress(float(lr_conf))

    with col2:
        st.subheader("KNN")
        st.write("Prediction:", "Real" if knn_pred else "Fake")
        st.progress(float(knn_conf))
    # Final decision
    st.subheader("Final Decision")

    if lr_acc > knn_acc:
        st.success(f"Final Prediction (Best Model - LR): {'Real' if lr_pred else 'Fake'}")
    else:
        st.success(f"Final Prediction (Best Model - KNN): {'Real' if knn_pred else 'Fake'}")