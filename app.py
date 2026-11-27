import time, pandas as pd, streamlit as st, torch, torch.nn as nn
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

MODEL_PATH, SAMPLES_DIR, IMG_SIZE, LOW_CONFIDENCE_THRESHOLD = Path("plant_disease_model.pth"), Path("samples"), 224, 60.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEAM = ["Monish M (1BM24AI090)", "Mohammed Fahim (1BM24AI088)", "Prajwal C V (1BM24AI113)"]
GUIDE = "Prof. Shridevi A Desai"

TREATMENTS = {
    "healthy": {"severity": "None", "about": "No visible disease symptoms detected.", "chemical": ["No chemical treatment required."], "organic": ["Continue routine care."], "prevention": ["Water at base.", "Maintain airflow.", "Weekly inspection."]},
    "anthracnose": {"severity": "Moderate to High", "about": "Fungal disease causing dark, sunken lesions.", "chemical": ["Spray copper oxychloride (0.3%) or mancozeb (0.25%).", "Carbendazim (0.1%)."], "organic": ["Neem oil spray (3-5 ml/L).", "Remove infected leaves."], "prevention": ["Prune canopy.", "Keep foliage dry."]},
    "powdery mildew": {"severity": "Moderate", "about": "White powdery coating on leaves.", "chemical": ["Wettable sulphur (0.2-0.3%).", "Hexaconazole (0.1%)."], "organic": ["Baking soda solution.", "Neem oil spray."], "prevention": ["Full sun.", "Avoid excess nitrogen."]},
    "leaf spot": {"severity": "Low to Moderate", "about": "Fungal/bacterial spots with yellow halos.", "chemical": ["Copper-based fungicide.", "Mancozeb (0.25%)."], "organic": ["Neem oil or garlic extract."], "prevention": ["Drip watering.", "Disinfect tools."]}
}
GENERIC_TREATMENT = {"severity": "Unknown", "about": "Detailed guidance logged in system.", "chemical": ["Consult local agricultural officer."], "organic": ["Neem oil spray."], "prevention": ["Keep foliage dry."]}

PREPROCESS = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists(): raise FileNotFoundError("Model missing")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint['classes']
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(checkpoint['model_state'])
    return model.to(DEVICE).eval(), class_names

try:
    model, CLASS_NAMES = load_model()
    model_loaded, load_error = True, None
except Exception as e:
    model_loaded, load_error = False, e
    CLASS_NAMES = []

def predict(model, image, k=3):
    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): probs = torch.softmax(model(tensor), dim=1)[0]
    top_p, top_i = torch.topk(probs, k=min(k, len(CLASS_NAMES)))
    return [(CLASS_NAMES[i], p * 100) for p, i in zip(top_p.tolist(), top_i.tolist())]

def timed_predict(model, image, k=3):
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    start = time.perf_counter()
    results = predict(model, image, k)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    return results, (time.perf_counter() - start) * 1000

def get_treatment(d):
    key = d.lower().strip()
    if key in TREATMENTS: return TREATMENTS[key]
    for name, info in TREATMENTS.items():
        if name in key: return info
    return GENERIC_TREATMENT

with st.sidebar:
    st.subheader("Model")
    st.markdown(f"**Architecture:** MobileNetV2\n**Resolution:** {IMG_SIZE}x{IMG_SIZE}\n**Classes:** {len(CLASS_NAMES) if model_loaded else 26}")
    st.divider()
    st.subheader("Team")
    for m in TEAM: st.markdown(f"- {m}")
    st.caption(f"Guide: {GUIDE}")

st.title("Plant Disease Detection")
if not model_loaded:
    st.error(f"Failed to load model: {load_error}")
    st.stop()

if "sample_path" not in st.session_state: st.session_state.sample_path = None
uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png", "webp"])

with st.expander("Sample images"):
    samples = sorted(SAMPLES_DIR.iterdir()) if SAMPLES_DIR.exists() else []
    for p in samples:
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and st.button(p.stem, key=f"s_{p.name}"):
            st.session_state.sample_path = str(p); st.rerun()

image = Image.open(uploaded) if uploaded else (Image.open(st.session_state.sample_path) if st.session_state.sample_path else None)
source_name = uploaded.name if uploaded else (f"Sample: {Path(st.session_state.sample_path).name}" if st.session_state.sample_path else None)

if not image: st.info("Please upload an image or select a sample to begin."); st.stop()

with st.spinner("Analyzing..."): results, latency_ms = timed_predict(model, image, k=3)
pred_class, top_conf = results[0]
treatment = get_treatment(pred_class)

with st.container(border=True):
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown(f"**Prediction:** {pred_class}")
    c2.markdown(f"**Confidence:** {top_conf:.2f}%")
    c3.markdown(f"**Latency:** {latency_ms:.1f} ms")
    st.markdown(f"**Severity:** {treatment['severity']}")
    if "healthy" not in pred_class.lower() and top_conf < LOW_CONFIDENCE_THRESHOLD:
        st.warning("Low confidence prediction.")

img_col, prob_col = st.columns([1, 1.3], gap="large")
with img_col:
    with st.container(border=True):
        st.image(image, caption=source_name, use_container_width=True)
with prob_col:
    with st.container(border=True):
        st.markdown("**Top-3 Predictions**")
        st.dataframe(pd.DataFrame(results, columns=["Class", "Confidence"]), hide_index=True, use_container_width=True, column_config={"Confidence": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

with st.container(border=True):
    st.markdown("**Remediation Guide**")
    st.write(treatment["about"])
    for tab, key in zip(st.tabs(["Chemical", "Organic", "Prevention"]), ("chemical", "organic", "prevention")):
        with tab:
            for step in treatment[key]: st.markdown(f"- {step}")

report = f"REPORT\nGenerated: {datetime.now():%Y-%m-%d %H:%M}\nImage: {source_name}\nPrediction: {pred_class} ({top_conf:.1f}%)\nLatency: {latency_ms:.1f}ms\n\nTop-3:\n" + "\n".join([f"- {l}: {c:.1f}%" for l, c in results])
st.download_button("Download Report", report, file_name="report.txt")