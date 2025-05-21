import streamlit as st

st.set_page_config(page_title="Animal or Plant Classifier", layout="centered")

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import tempfile
import os

MODEL_PATH = r"C:\Users\Angelo\Desktop\CPE 019 Final Project\best_model_classifier.pth"
CLASS_NAMES = ["animals", "plants"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

st.title("🐾 Animal or 🌱 Plant Classifier")
st.markdown("Upload an image of an animal or a plant.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_percent = confidence.item() * 100

    st.markdown(f"### Prediction: **{predicted_class.capitalize()}**")
    st.markdown(f"**Confidence:** {confidence_percent:.2f}%")