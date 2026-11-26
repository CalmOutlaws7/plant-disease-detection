from datetime import datetime
from pathlib import Path
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models,transforms
MODEL_PATH=Path("plant_disease_model.pth")
SAMPLES_DIR=Path("samples")
IMG_SIZE=224
LOW_CONFIDENCE_THRESHOLD=60.0
PROJECT_INFO={"title":"Plant Disease Detection","team":["Monish M (1BM24AI090)","Mohammed Fahim (1BM24AI088)","Prajwal C V (1BM24AI113)"],"guide":"Prof. Shridevi A Desai","college":"BMS College of Engineering","department":"AI & ML","architecture":"MobileNetV2 (Transfer Learning)","dataset_size":"15,000+ images","plants":"Roxburgh Fig, Jasmine, Mango, Mint, Drumstick","accuracy":"~95.77%","hardware":"NVIDIA RTX 5060","framework":"PyTorch + Streamlit"}
TREATMENTS={"healthy":{"severity":"None","about":"No visible disease symptoms detected. The leaf appears healthy.","chemical":["No chemical treatment required."],"organic":["Continue routine care; avoid unnecessary sprays."],"prevention":["Water at the base of the plant, not over the leaves.","Maintain good spacing and airflow; prune crowded branches.","Inspect leaves weekly for early symptoms.","Use balanced fertilization; avoid excess nitrogen."]},"anthracnose":{"severity":"Moderate to High","about":"Fungal disease causing dark, sunken lesions on leaves, flowers and fruits, worst in warm, humid weather.","chemical":["Spray copper oxychloride (0.3%) or mancozeb (0.25%).","Systemic option: carbendazim (0.1%) or azoxystrobin.","Repeat at 10-15 day intervals during humid periods."],"organic":["Neem oil spray (3-5 ml/L water) every 7-10 days.","Bordeaux mixture (1%) as a protective spray.","Remove and burn badly infected leaves and twigs."],"prevention":["Prune to open the canopy and improve air circulation.","Avoid overhead irrigation; keep foliage dry.","Clear fallen leaves and debris from around the plant."]},"powdery mildew":{"severity":"Moderate","about":"Fungal disease producing a white, powdery coating on leaves and shoots; reduces photosynthesis and causes leaf curl.","chemical":["Wettable sulphur (0.2-0.3%) spray.","Alternatives: hexaconazole (0.1%) or dinocap.","Repeat after 10-14 days if symptoms continue."],"organic":["Baking soda spray: 1 tsp baking soda + few drops of liquid soap per litre.","Neem oil spray (3-5 ml/L) every 7 days.","Remove and destroy heavily affected leaves."],"prevention":["Ensure full sun and good airflow around plants.","Avoid excessive nitrogen fertilizer.","Water early in the day so leaves dry quickly."]},"leaf spot":{"severity":"Low to Moderate","about":"Fungal or bacterial infection causing brown/black spots, often with yellow halos, that can merge and cause leaf drop.","chemical":["Copper-based fungicide (copper oxychloride 0.3%).","Mancozeb (0.25%) spray for fungal leaf spots.","Repeat every 10-14 days until new growth is clean."],"organic":["Neem oil or garlic-chilli extract spray.","Pick off and dispose of spotted leaves."],"prevention":["Avoid wetting the foliage; use drip or base watering.","Disinfect pruning tools between plants.","Clear plant debris at season end."]}}
GENERIC_TREATMENT={"severity":"Unknown","about":"Detailed guidance for this specific condition is logged in system records.","chemical":["Consult local agricultural extension officer."],"organic":["Neem oil spray and removal of affected leaves."],"prevention":["Keep foliage dry and maintain proper canopy airflow."]}
st.set_page_config(page_title="Plant Disease Detector",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>.block-container{padding-top:1.5rem;padding-bottom:1.5rem;}.hero{background:#1b5e20;padding:1.2rem 1.5rem;border-radius:6px;color:white;margin-bottom:1rem;}.hero h1{margin:0;font-size:1.8rem;color:white;}.hero p{margin:.2rem 0 0 0;opacity:.9;}</style>""",unsafe_allow_html=True)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESS=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists():raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")
    checkpoint=torch.load(MODEL_PATH,map_location=DEVICE,weights_only=False)
    class_names=checkpoint['classes']
    state_dict=checkpoint['model_state']
    model=models.mobilenet_v2(weights=None)
    model.classifier[1]=nn.Linear(model.last_channel,len(class_names))
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval(),class_names
try:
    model,CLASS_NAMES=load_model()
    model_loaded=True
except Exception as e:
    model_loaded=False
    load_error=e
def predict(model:nn.Module,image:Image.Image,k:int=3):
    tensor=PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():probs=torch.softmax(model(tensor),dim=1)[0]
    top_p,top_i=torch.topk(probs,k=min(k,len(CLASS_NAMES)))
    return [(CLASS_NAMES[i],p*100) for p,i in zip(top_p.tolist(),top_i.tolist())]
def get_treatment(disease:str)->dict:
    key=disease.lower().strip()
    if key in TREATMENTS:return TREATMENTS[key]
    for name,info in TREATMENTS.items():
        if name in key:return info
    return GENERIC_TREATMENT
def get_samples():
    if not SAMPLES_DIR.exists():return []
    exts={".jpg",".jpeg",".png",".webp"}
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in exts)
with st.sidebar:
    st.markdown("## Project Info")
    st.caption(PROJECT_INFO["title"])
    with st.expander("Team",expanded=True):
        for member in PROJECT_INFO["team"]:st.markdown(f"- {member}")
        st.caption(f"Guide: {PROJECT_INFO['guide']}")
        st.caption(f"{PROJECT_INFO['college']} | {PROJECT_INFO['department']}")
    with st.expander("Model & Data",expanded=True):
        st.markdown(f"Architecture: {PROJECT_INFO['architecture']}")
        st.markdown(f"Dataset: {PROJECT_INFO['dataset_size']}")
        st.markdown(f"Accuracy: {PROJECT_INFO['accuracy']}")
        st.markdown(f"Plants: {PROJECT_INFO['plants']}")
        if model_loaded:st.markdown(f"Loaded Classes: {len(CLASS_NAMES)}")
    with st.expander("System",expanded=True):
        st.markdown(f"Training HW: {PROJECT_INFO['hardware']}")
        st.markdown(f"Framework: {PROJECT_INFO['framework']}")
        st.markdown(f"Inference device: `{DEVICE.type.upper()}`")
    st.divider()
    st.caption("AAT Review Tool")
st.markdown("""<div class="hero"><h1>Plant Disease Detection Dashboard</h1><p>Upload a leaf image to classify health condition and view remediation steps.</p></div>""",unsafe_allow_html=True)
if not model_loaded:
    st.error(f"Could not load the model: {load_error}")
    st.stop()
if "sample_path" not in st.session_state:st.session_state.sample_path=None
uploaded=st.file_uploader("Choose a leaf image (JPG, PNG)...",type=["jpg","jpeg","png","webp"])
with st.expander("Test with Sample Images",expanded=False):
    samples=get_samples()
    if not samples:st.info("Place sample images in samples/ folder.")
    else:
        for path in samples:
            if st.button(path.stem):
                st.session_state.sample_path=str(path)
                st.rerun()
image,source_name=None,None
if uploaded is not None:image,source_name=Image.open(uploaded),uploaded.name
elif st.session_state.sample_path:
    image=Image.open(st.session_state.sample_path)
    source_name=f"Sample: {Path(st.session_state.sample_path).name}"
if image is None:
    st.info("Upload a leaf image above to begin diagnosis.")
else:
    st.divider()
    with st.spinner("Analyzing leaf patterns..."):results=predict(model,image,k=3)
    top_label,top_conf=results[0]
    predicted_class=top_label
    treatment=get_treatment(predicted_class)
    is_healthy="healthy" in predicted_class.lower()
    img_col,res_col=st.columns([1,1.3],gap="large")
    with img_col:
        st.subheader("Leaf Image")
        st.image(image,caption=source_name,use_container_width=True)
    with res_col:
        st.subheader("Diagnosis Results")
        st.success(f"Detected Class: `{predicted_class}`")
        st.metric(label="Confidence Score",value=f"{top_conf:.2f}%")
        if is_healthy:st.info("Status: Healthy Specimen")
        elif top_conf<LOW_CONFIDENCE_THRESHOLD:st.warning("Status: Low Confidence")
        else:st.error(f"Status: Condition Detected | Severity: {treatment['severity']}")
        st.markdown("#### Top-3 Predictions")
        for label,conf in results:
            c1,c2=st.columns([3,1])
            c1.markdown(f"- {label}")
            c2.markdown(f"{conf:.1f}%")
            st.progress(min(max(conf/100,0.0),1.0))
    st.divider()
    st.subheader("Remediation & Treatment Guide")
    st.write(treatment["about"])
    t1,t2,t3=st.tabs(["Chemical Control","Organic Remedies","Prevention"])
    with t1:
        for step in treatment["chemical"]:st.markdown(f"- {step}")
    with t2:
        for step in treatment["organic"]:st.markdown(f"- {step}")
    with t3:
        for step in treatment["prevention"]:st.markdown(f"- {step}")
    report="\n".join(["PLANT DISEASE DETECTION REPORT",f"Generated: {datetime.now():%Y-%m-%d %H:%M}",f"Image: {source_name}",f"Primary Prediction: {predicted_class} ({top_conf:.1f}%)","","Top-3 Predictions:",*[f"  - {l}: {c:.1f}%" for l,c in results],])
    st.download_button("Download Diagnosis Report (.txt)",report,file_name="plant_diagnosis_report.txt")
                     
