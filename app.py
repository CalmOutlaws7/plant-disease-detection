import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

CLASSES = [
    'Drumstick Healthy', 'Drumstick Diseased',
    'Jasmine Healthy', 'Jasmine Diseased',
    'Mango Healthy', 'Mango Diseased',
    'Mint Healthy', 'Mint Diseased',
    'Roxburgh Fig Healthy', 'Roxburgh Fig Diseased'
]

@st.cache_resource
def load_model():
    model = torch.load('model.pth', map_location='cpu')
    model.eval()
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

st.title("Plant Disease Detection System")
st.write("A deep learning application for classifying plant leaf health and diseases.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if model_loaded:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        if st.button('Predict Disease'):
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            st.write(f"Prediction: Class {predicted_idx.item()}")
            st.write(f"Confidence: {confidence.item() * 100:.2f}%")
    else:
        st.error("Model file could not be loaded.")