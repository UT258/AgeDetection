import streamlit as st
import cv2
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import numpy as np

# Load model and feature extractor
@st.cache_resource
def load_model():
    extractor = AutoFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")
    model = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
    return extractor, model

extractor, model = load_model()

st.title("🎥 Age Detection with Webcam + Hugging Face")
st.write("Click the button to capture an image from your webcam and detect the age group.")

# Webcam Capture
def capture_image():
    cap = cv2.VideoCapture(0)
    st.info("📸 Press 's' to capture the image or 'q' to quit.")
    
    captured = False
    img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to grab frame from webcam.")
            break

        cv2.imshow("Webcam - Press 's' to capture", frame)
        key = cv2.waitKey(1)

        if key == ord("s"):
            img = frame
            captured = True
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return img if captured else None

# Button to capture image
if st.button("📷 Capture from Webcam"):
    captured_img = capture_image()
    if captured_img is not None:
        # Convert to PIL Image
        img_rgb = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        st.image(pil_image, caption="Captured Image", use_column_width=True)

        # Predict using Hugging Face model
        inputs = extractor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs).item()
            age_group = model.config.id2label[predicted_class]

        st.success(f"🧠 Predicted Age Group: **{age_group}**")
    else:
        st.warning("No image captured.")
