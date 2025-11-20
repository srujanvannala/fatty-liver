import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# ----------------------------
# Load trained model
# ----------------------------
from train_model import load_trained_model

try:
    model = load_trained_model()
except FileNotFoundError:
    st.error("Trained model 'liver_model.h5' not found. Please train the model first using train_model.py.")
    st.stop()

# ----------------------------
# Preprocess image
# ----------------------------
def preprocess_image(img, img_size=(224,224)):
    img = img.resize(img_size)
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, H, W, 3)
    return img_array

# ----------------------------
# Mapping of classes
# ----------------------------
DATA_DIR = "data"
classes = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
classes.sort()  # ensure consistent order
st.write("Detected classes in dataset:", classes)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Liver Ultrasound Classifier", layout="wide")
st.title("📸 Liver Ultrasound Image Classifier")
st.write("Upload an ultrasound image of liver and get prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=400)

    if st.button("Predict"):
        # preprocess
        processed_img = preprocess_image(img)
        # predict
        prediction = model.predict(processed_img)
        predicted_index = np.argmax(prediction)
        predicted_label = classes[predicted_index]

        st.success(f"### 🩺 Predicted Class: **{predicted_label}**")
        st.write("Prediction Probabilities:")
        for i, cls in enumerate(classes):
            st.write(f"{cls}: {prediction[0][i]*100:.2f}%")
