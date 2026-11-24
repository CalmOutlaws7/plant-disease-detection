import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

CLASSES = ['Drumstick Diseased', 'Drumstick Healthy','Jasmine Diseased', 'Jasmine Healthy','Mango Diseased', 'Mango Healthy','Mint Diseased', 'Mint Healthy',
    'Roxburgh Fig Diseased', 'Roxburgh Fig Healthy']

TREATMENTS = {'Drumstick Diseased': 'Apply neem oil spray and ensure proper soil drainage.','Jasmine Diseased': 'Remove infected leaves and apply organic fungicide.','Mango Diseased': 'Use copper-based fungicide and prune affected branches.',
    'Mint Diseased': 'Improve air circulation and avoid overwatering.','Roxburgh Fig Diseased': 'Clear fallen debris and apply bio-pesticide.'}

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(CLASSES))
    model.load_state_dict(torch.load('model.pth', map_location='torch.device("cpu")'))
    model.eval()
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

st.title("Plant Disease Detection System")
st.write("A deep learning web application for classifying plant leaf health and viewing remediation steps.")

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
        
        if st.button('Analyze Image'):
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top_prob, top_cat = torch.topk(probabilities, 3)
                
            st.subheader("Top Predictions")
            for i in range(3):
                class_name = CLASSES[top_cat[i].item()]
                score = top_prob[i].item() * 100
                st.write(f"{i+1}. {class_name}: {score:.2f}%")
            
            primary_prediction = CLASSES[top_cat[0].item()]
            st.markdown("---")
            st.subheader("Recommendation")
            
            if "Diseased" in primary_prediction:
                treatment = TREATMENTS.get(primary_prediction, 'Apply standard organic pest control and isolate plant.')
                st.warning(f"Status: Diseased\n\nRecommended Action: {treatment}")
            else:
                st.success("Status: Healthy\n\nNo treatment required. Maintain standard plant care routine.")
    else:
        st.error("Model file could not be loaded. Please ensure model.pth is present.")