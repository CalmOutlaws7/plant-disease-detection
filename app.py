import time, pandas as pd, streamlit as st, torch, torch.nn as nn
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

MODEL_PATH, SAMPLES_DIR, IMG_SIZE = Path("plant_disease_model.pth"), Path("samples"), 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEAM = ["Monish M (1BM24AI090)", "Mohammed Fahim (1BM24AI088)", "Prajwal C V (1BM24AI113)"]
GUIDE = "Prof. Shridevi A Desai"

PREPROCESS = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
st.set_page_config(page_title="Leaf Pathology Classifier", layout="wide")

st.markdown("""<style>
.stApp { background-color: #0b0f19; color: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif; }
#MainMenu, footer { visibility: hidden; }
.glass { background: rgba(255,255,255,0.03); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 20px; margin-bottom: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.37); }
.stButton>button { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); color: #f3f4f6; border-radius: 8px; }
.stButton>button:hover { background: rgba(255,255,255,0.12); border-color: rgba(255,255,255,0.2); }
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists(): raise FileNotFoundError("Model missing")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = ckpt['classes']
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, len(classes))
    m.load_state_dict(ckpt['model_state'])
    return m.to(DEVICE).eval(), classes

try:
    model, CLASS_NAMES = load_model()
    model_loaded, load_error = True, None
except Exception as e:
    model_loaded, load_error, CLASS_NAMES = False, e, []

def run_inference(img, k=5):
    x = PREPROCESS(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    top_p, top_i = torch.topk(probs, k=min(k, len(CLASS_NAMES)))
    return [(CLASS_NAMES[i], float(p * 100), float(logits[0, i])) for p, i in zip(top_p, top_i)], ms

with st.sidebar:
    st.markdown("### Spec")
    st.text(f"Model: MobileNetV2\nDevice: {DEVICE.type.upper()}\nClasses: {len(CLASS_NAMES)}")
    st.divider()
    st.markdown("### Team")
    st.text("\n".join(TEAM) + f"\n\nGuide: {GUIDE}")

st.markdown("## Leaf Pathology Inference")
if not model_loaded: st.error(f"Error: {load_error}"); st.stop()

if "sample_path" not in st.session_state: st.session_state.sample_path = None
c_up, c_samp = st.columns([2, 1])
with c_up: uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
with c_samp:
    samples = sorted([p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]) if SAMPLES_DIR.exists() else []
    if samples:
        choice = st.selectbox("Sample", ["None"] + [p.name for p in samples], label_visibility="collapsed")
        if choice != "None": st.session_state.sample_path = str(SAMPLES_DIR / choice)

image = Image.open(uploaded) if uploaded else (Image.open(st.session_state.sample_path) if st.session_state.sample_path else None)
source_name = uploaded.name if uploaded else (Path(st.session_state.sample_path).name if st.session_state.sample_path else None)

if not image: st.info("Awaiting input image tensor..."); st.stop()

results, latency_ms = run_inference(image, k=5)
top_class, top_conf, top_logit = results[0]

c_img, c_res = st.columns([1, 1.4], gap="large")
with c_img:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.caption(f"`{source_name}` ({image.width}x{image.height}px)")
    st.markdown('</div>', unsafe_allow_html=True)

with c_res:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(f"### `{top_class}`")
    st.caption(f"Conf: **{top_conf:.2f}%** | Logit: **{top_logit:.3f}** | Latency: **{latency_ms:.1f}ms**")
    st.divider()
    st.markdown("**Top-5 Probability Distribution**")
    df = pd.DataFrame(results, columns=["Class Label", "Probability (%)", "Logit"])
    df["Probability (%)"] = df["Probability (%)"].map(lambda v: f"{v:.2f}%")
    df["Logit"] = df["Logit"].map(lambda v: f"{v:.3f}")
    st.dataframe(df, hide_index=True, use_container_width=True)
    log_txt = f"timestamp: {datetime.now().isoformat()}\nfile: {source_name}\nlatency_ms: {latency_ms:.2f}\n\n" + "\n".join([f"{c:<30} {p:>6.2f}%" for c, p, _ in results])
    st.download_button("Export Telemetry Log", log_txt, file_name="telemetry.txt")
    st.markdown('</div>', unsafe_allow_html=True)