import streamlit as st
import cv2
import numpy as np
import joblib
import os
import random
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Deepfake Detection", layout="wide")

# -----------------------
# UI Styling
# -----------------------
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; padding: 40px 0;'>
    <h1 style='font-size:40px; color:#00E5FF;'>Deepfake Detection System</h1>
    <p style='font-size:18px; color:#bbb;'>AI-powered system to detect real vs fake images</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Load Models
# -----------------------
lr = joblib.load("lr_model.pkl")
knn = joblib.load("knn_model.pkl")

lr_acc = joblib.load("lr_acc.pkl")
knn_acc = joblib.load("knn_acc.pkl")

# -----------------------
# Overview Section
# -----------------------
st.subheader("System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{lr_acc*100:.1f}%")

with col2:
    st.metric("Models", "2")

with col3:
    st.metric("Dataset Size", "10K Images")

st.markdown("---")

# -----------------------
# Model Comparison Graph
# -----------------------
st.subheader("Model Comparison")

fig, ax = plt.subplots()
ax.bar(["Logistic Regression", "KNN"], [lr_acc, knn_acc])
ax.set_ylim(0, 1)

for i, v in enumerate([lr_acc, knn_acc]):
    ax.text(i, v + 0.02, f"{v:.2f}", ha='center')

st.pyplot(fig)

st.markdown("---")

# -----------------------
# Conclusion
# -----------------------
st.subheader("Conclusion")

if lr_acc > knn_acc:
    st.success("Logistic Regression performs better for this dataset")
else:
    st.success("KNN performs better for this dataset")

st.markdown("---")

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
# Upload Image Section
# -----------------------
st.subheader("Upload Image for Detection")

uploaded = st.file_uploader("Choose image", type=["jpg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    features = preprocess(img)

    lr_pred = lr.predict(features)[0]
    knn_pred = knn.predict(features)[0]

    lr_prob = lr.predict_proba(features)[0]
    knn_prob = knn.predict_proba(features)[0]

    lr_conf = max(lr_prob)
    knn_conf = max(knn_prob)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### Model Predictions")

        st.write(f"Logistic Regression: {'Real' if lr_pred else 'Fake'} ({lr_conf*100:.1f}%)")
        st.progress(float(lr_conf))

        st.write(f"KNN: {'Real' if knn_pred else 'Fake'} ({knn_conf*100:.1f}%)")
        st.progress(float(knn_conf))

    # Final Result
    st.markdown("### Final Result")

    if lr_acc > knn_acc:
        final = "Real" if lr_pred else "Fake"
    else:
        final = "Real" if knn_pred else "Fake"

    if final == "Real":
        st.success("REAL IMAGE DETECTED")
    else:
        st.error("FAKE IMAGE DETECTED")

st.markdown("---")

# -----------------------
# Random Image Test
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

    lr_conf = max(lr.predict_proba(features)[0])
    knn_conf = max(knn.predict_proba(features)[0])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Logistic Regression")
        st.write("Prediction:", "Real" if lr_pred else "Fake")
        st.progress(float(lr_conf))

    with col2:
        st.subheader("KNN")
        st.write("Prediction:", "Real" if knn_pred else "Fake")
        st.progress(float(knn_conf))

    st.subheader("Final Decision")

    if lr_acc > knn_acc:
        st.success(f"Final Prediction (LR): {'Real' if lr_pred else 'Fake'}")
    else:
        st.success(f"Final Prediction (KNN): {'Real' if knn_pred else 'Fake'}")