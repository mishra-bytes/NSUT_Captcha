import streamlit as st
import numpy as np
import cv2
import os
import zipfile
import io
import shutil
import time
import base64
import random
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Import our custom modules
import backend
import training_utils
import time
import random

def rotate_fun_facts(fact_box, facts, duration_seconds=2.5, cycles=3):
    """
    Displays a new random fun fact every `duration_seconds`.
    Runs for `cycles` iterations.
    
    fact_box : Streamlit placeholder (st.empty())
    facts    : list of strings
    """
    for _ in range(cycles):
        fact = random.choice(facts)
        fact_box.markdown(
            f'<div class="fun-fact-container"><span class="fun-fact-text">{fact}</span></div>',
            unsafe_allow_html=True
        )
        time.sleep(duration_seconds)
    # Clear after rotation
    fact_box.empty()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CAPTCHArd",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL GLASSMORPHISM THEME ---
st.markdown("""
<style>
    /* FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    :root {
        --primary: #2563EB;
        --primary-hover: #1D4ED8;
        --text-main: #1E293B;
        --text-sub: #64748B;
        --glass-bg: rgba(255, 255, 255, 0.65);
        --glass-border: 1px solid rgba(255, 255, 255, 0.9);
        --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
    }

    /* FULL BLURRED BACKGROUND */
    .stApp {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        background-attachment: fixed;
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.6);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    /* GLASS CARDS - ROBUST */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: var(--glass-border);
        border-radius: 16px;
        padding: 24px;
        box-shadow: var(--glass-shadow);
        margin-bottom: 0px; /* Reset margin to prevent ghost spacing */
        height: 100%;
    }

    /* TYPOGRAPHY */
    .section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-sub);
        font-weight: 700;
        margin-bottom: 1rem;
    }

    /* BUTTONS (Scoped to avoid breaking File Uploader) */
    div.stButton > button {
        background-color: var(--primary);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
        width: 100%;
        height: 50px;
    }
    
    div.stButton > button:hover {
        background-color: var(--primary-hover);
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* FIX FOR FILE UPLOADER BUTTON */
    [data-testid="stFileUploader"] button {
        width: auto !important;
        height: auto !important;
        background-color: transparent !important;
        color: var(--text-main) !important;
        border: 1px solid rgba(0,0,0,0.2) !important;
        box-shadow: none !important;
        transform: none !important;
        padding: 0.25rem 0.75rem !important;
        margin-top: 0px !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background-color: rgba(0,0,0,0.05) !important;
        border-color: var(--primary) !important;
        color: var(--primary) !important;
    }

    /* PREDICTION BOX */
    .prediction-display {
        background: rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 10px;
    }
    
    .prediction-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem; /* Reduced for single line fit */
        font-weight: 700;
        color: var(--primary);
        letter-spacing: 0.2rem;
        line-height: 1;
        margin: 10px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .confidence-tag {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* IMAGES */
    img {
        border-radius: 8px;
    }
    
    /* INPUTS */
    div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.1);
    }

    /* GRID FOR IMAGES */
    .visual-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 24px;
    }
    
    .digit-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
    }
    
    /* FUN FACT BLINKING */
    @keyframes blink-animation {
      0% { opacity: 0; }
      10% { opacity: 1; }
      90% { opacity: 1; }
      100% { opacity: 0; }
    }
    
    .fun-fact-container {
        padding: 15px 20px;
        text-align: center;
        background: rgba(37, 99, 235, 0.05);
        border: 1px solid rgba(37, 99, 235, 0.1);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .fun-fact-text {
        color: #2563EB;
        font-weight: 500;
        font-size: 0.95rem;
        animation: blink-animation 2.5s infinite; 
    }

</style>
""", unsafe_allow_html=True)

# --- HELPER: CUSTOM LOADER (uses st.image for smoother behavior) ---
def render_custom_loader():
    file_path = "assets/loading.gif"
    if os.path.exists(file_path):
        try:
            st.image(file_path, width=40)
        except Exception:
            # fallback to base64 approach if direct image load fails
            with open(file_path, "rb") as f:
                data_url = base64.b64encode(f.read()).decode("utf-8")
            st.markdown(
                f'<div style="display: flex; justify-content: center; padding: 2rem;"><img src="data:image/gif;base64,{data_url}" width="40"></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Processing...")

# --- HELPER: IMAGE TO HTML ---
def get_image_html(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64}"

# --- HELPER: LOG EXPERIMENT ---
def log_experiment(method, epochs, accuracy, details=""):
    log_file = "training_history.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_entry = {
        "Timestamp": timestamp,
        "Method": method,
        "Epochs": epochs,
        "Val Accuracy": f"{accuracy:.4f}",
        "Details": details
    }
    
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
        
    df.to_csv(log_file, index=False)

# --- FUN FACTS (no emojis) ---
FUN_FACTS = [
    "Convolutional Neural Networks analyze images by sliding filters over them.",
    "Bayesian Optimization builds a probability model to find the best parameters efficiently.",
    "Max Pooling reduces image dimensions to highlight the most important features.",
    "Dropout randomly ignores neurons during training to prevent overfitting.",
    "The Learning Rate controls how much the model updates its weights each step.",
    "One Epoch means the model has seen the entire training dataset once.",
    "Softmax function converts raw model outputs into probabilities.",
    "ReLU activation helps the network learn complex patterns by adding non-linearity."
]

# --- SAFE ZIP EXTRACT (protect against ZipSlip) ---
def safe_extract(zip_file: zipfile.ZipFile, target_dir: str):
    """
    Safe ZIP extraction that handles:
    - ZipSlip attacks
    - Folder/file name collisions on Windows
    - Directories inside ZIP
    """
    for member in zip_file.infolist():

        # Normalize the path
        normalized = os.path.normpath(member.filename)

        # Security: avoid paths escaping target_dir
        if normalized.startswith("..") or os.path.isabs(normalized):
            continue

        # Skip MacOS junk
        if normalized.startswith("__MACOSX"):
            continue

        dest_path = os.path.join(target_dir, normalized)

        # If the item is a directory, just create it and continue
        if member.is_dir():
            os.makedirs(dest_path, exist_ok=True)
            continue

        # Ensure parent directory exists
        parent_dir = os.path.dirname(dest_path)
        os.makedirs(parent_dir, exist_ok=True)

        # Extract the file safely
        with zip_file.open(member) as source, open(dest_path, "wb") as target:
            shutil.copyfileobj(source, target)

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists("./assets/full_logo_colour.png"):
        st.image("./assets/full_logo_colour.png", use_container_width=True)
    elif os.path.exists("./assets/logo.png"):
        st.image("./assets/logo.png", width=60)
        st.markdown("### CAPTCHArd")
    else:
        st.markdown("### CAPTCHArd")
    
    st.markdown(" ")
    st.markdown('<p class="section-title" style="margin-bottom: 0.5rem;">Platform</p>', unsafe_allow_html=True)
    mode = st.radio("Platform", ["Live Dashboard", "Training Studio"], label_visibility="collapsed")
    
    st.write("") 
    st.caption("v3.0 Enterprise Edition")
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748B;">
            Made with <span style="color: #E25555;">&hearts;</span> by <span style="font-weight: 600;">Aditya Mishra</span>
        </div>
    """, unsafe_allow_html=True)

# --- INIT SESSION & CACHED MODEL LOADER ---
@st.cache_resource
def cached_load_pretrained_model():
    return backend.load_pretrained_model()

if 'model' not in st.session_state or st.session_state.get('model') is None:
    try:
        st.session_state.model = cached_load_pretrained_model()
    except Exception:
        # If model load fails, set to None so UI shows offline.
        st.session_state.model = None

if 'dataset_uploaded' not in st.session_state:
    st.session_state.dataset_uploaded = False

# --- UTILS ---
def load_uploaded_dataset(uploaded_file):
    """
    Expects a Streamlit UploadedFile (zip). Returns (data_array, labels_array, valid_count).
    """
    # Create temp dir
    temp_dir = "temp_dataset"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Use safe_extract to avoid ZipSlip
    with zipfile.ZipFile(uploaded_file, 'r') as z:
        safe_extract(z, temp_dir)

    data, labels = [], []
    valid_count = 0
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                label_text = os.path.splitext(f)[0]
                if len(label_text) != 5 or not label_text.isdigit():
                    continue
                path = os.path.join(root, f)
                # backend.preprocess_captcha_v2 previously accepted path or bytes; your original used path
                _, cleaned = backend.preprocess_captcha_v2(path)
                digits = backend.segment_characters_robust(cleaned)
                if len(digits) == 5:
                    valid_count += 1
                    for i, d in enumerate(digits):
                        img = d / 255.0
                        img = np.expand_dims(img, axis=-1)
                        data.append(img)
                        labels.append(int(label_text[i]))
    # Cleanup temp if desired (keep for debugging)
    # shutil.rmtree(temp_dir)
    return np.array(data), np.array(labels), valid_count

def save_and_update_model(model):
    try:
        os.makedirs('model', exist_ok=True)
        path = os.path.join('model', 'final_captcha_model.h5')
        model.save(path)
        st.session_state.model = model
        st.toast("Model saved successfully")
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

# =========================================================
#  MODE 1: LIVE DASHBOARD
# =========================================================
if mode == "Live Dashboard":
    
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.5rem; font-weight: 700;">Live Inference Dashboard</h1>
        <p style="color: #64748B;">Real-time computer vision pipeline for security analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- LAYOUT: Left (70%) for Status/Visuals, Right (30%) for Controls/Results ---
    col_main, col_sidebar = st.columns([7, 3], gap="large")
    
    # === LEFT COLUMN: MONITORING & VISUALS ===
    with col_main:
        # System Status
        status_content = ""
        if st.session_state.model:
            status_content = '<span style="color: #166534; font-weight: 600;">● Online</span>: Model loaded and ready for inference.'
        else:
            status_content = '<span style="color: #DC2626; font-weight: 600;">● Offline</span>: No model detected.'
            
        st.markdown(f"""
<div class="glass-card" style="padding: 1.5rem; display: flex; align-items: center; min-height: 80px;">
    <div>
        <span class="section-title" style="margin-right: 15px; display: block; margin-bottom: 5px;">System Status</span>
        <span style="font-size: 1rem; color: #1E293B;">{status_content}</span>
    </div>
</div>
""", unsafe_allow_html=True)
        
        visual_placeholder = st.empty()

    # === RIGHT COLUMN: ACTIONS & RESULTS ===
    with col_sidebar:
        st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
        if st.button("Fetch Captcha", use_container_width=True):
            st.session_state.trigger_fetch = True

        result_placeholder = st.empty()

    # --- LOGIC ---
    if st.session_state.get('trigger_fetch'):
        with visual_placeholder.container():
            with st.spinner("Processing..."):
                time.sleep(0.1) 
                fetcher = backend.CaptchaFetcher()
                img_bytes, error = fetcher.fetch_single_image()
                time.sleep(0.5) 

        if error:
            st.error(f"Connection Error: {error}")
        else:
            nparr = np.frombuffer(img_bytes, np.uint8)
            original_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            _, cleaned = backend.preprocess_captcha_v2(io.BytesIO(img_bytes))
            
            digits = backend.segment_characters_robust(cleaned)
            
            src_html = get_image_html(original_img)
            bin_html = get_image_html(cleaned)
            
            digits_divs = ""
            if len(digits) == 5:
                for d in digits:
                    digits_divs += f'<div style="text-align: center;"><img src="{get_image_html(d)}" style="width: 100%; border-radius: 8px; border: 1px solid rgba(0,0,0,0.05); box-shadow: 0 2px 4px rgba(0,0,0,0.05);"></div>'
            
            with visual_placeholder.container():
                st.markdown(f"""
<div class="glass-card">
    <p class="section-title">Visual Analysis</p>
    <div class="visual-grid">
        <div>
            <div style="font-size: 0.75rem; color: #64748B; margin-bottom: 8px; font-weight: 600; text-transform: uppercase;">Raw Input</div>
            <img src="{src_html}" style="width: 100%; border-radius: 12px; border: 1px solid rgba(0,0,0,0.05);">
        </div>
        <div>
            <div style="font-size: 0.75rem; color: #64748B; margin-bottom: 8px; font-weight: 600; text-transform: uppercase;">Binary Mask</div>
            <img src="{bin_html}" style="width: 100%; border-radius: 12px; border: 1px solid rgba(0,0,0,0.05);">
        </div>
    </div>
    <div style="border-top: 1px solid rgba(0,0,0,0.05); margin: 24px 0;"></div>
    <p class="section-title" style="margin-bottom: 15px;">Segmentation Stream</p>
    <div class="digit-grid">
        {digits_divs}
    </div>
</div>
""", unsafe_allow_html=True)
            
            if len(digits) == 5 and st.session_state.model:
                prediction = backend.predict_sequence(st.session_state.model, digits)
                st.session_state.last_prediction = prediction
            else:
                st.session_state.last_prediction = None

    if st.session_state.get('trigger_fetch'):
        with result_placeholder.container():
            st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
            
            if st.session_state.get('last_prediction'):
                st.markdown(f"""
<div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
    <p class="section-title">Inference Result</p>
    <div class="prediction-display">
        <div class="prediction-text">{st.session_state.last_prediction}</div>
        <div class="confidence-tag">High Confidence</div>
    </div>
</div>
""", unsafe_allow_html=True)
            elif 'last_prediction' in st.session_state and st.session_state.last_prediction is None:
                 st.warning("Segmentation Failed")
    else:
        with visual_placeholder.container():
             pass


# =========================================================
#  MODE 2: TRAINING STUDIO
# =========================================================
elif mode == "Training Studio":
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.5rem; font-weight: 700;">Training Studio</h1>
        <p style="color: #64748B;">Design, train, and optimize CNN architectures.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
<div class="glass-card" style="padding-bottom: 10px; margin-bottom: 20px;">
    <p class="section-title" style="margin-bottom: 5px;">Data Ingestion</p>
    <p style="font-size: 0.9rem; color: #64748B;">Upload your labeled dataset to begin.</p>
</div>
""", unsafe_allow_html=True)

    col_upload, col_stats = st.columns([2, 1])
    
    with col_upload:
        st.markdown('<div style="height: 5px;"></div>', unsafe_allow_html=True)
        upload = st.file_uploader("Upload Dataset (ZIP)", type="zip", label_visibility="collapsed")
    
    with col_stats:
        if upload and not st.session_state.dataset_uploaded:
            # Upload size check (limit: 200 MB)
            try:
                size_bytes = upload.size
            except Exception:
                size_bytes = None
            if size_bytes is not None and size_bytes > 200 * 1024 * 1024:
                st.error("File too large. Please upload a ZIP smaller than 200 MB.")
            else:
                with st.spinner("Processing..."):
                    X, y, count = load_uploaded_dataset(upload)
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.dataset_uploaded = True
                    st.session_state.sample_count = count
                    
        if st.session_state.dataset_uploaded:
             st.success(f"Ready: {st.session_state.sample_count} Samples")

    if st.session_state.dataset_uploaded:
        X, y = st.session_state.X, st.session_state.y
        y_encoded = to_categorical(y, 10)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Manual Tuning", "Bayesian Optimization"])
        
        with tab1:
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            with st.container():
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Convolution**")
                    f1 = st.slider("Layer 1 Filters", 16, 64, 32, 16)
                    f2 = st.slider("Layer 2 Filters", 32, 128, 64, 32)
                with c2:
                    st.markdown("**Dense**")
                    dense = st.select_slider("Neurons", options=[32, 64, 128, 256, 512], value=64)
                    dropout = st.slider("Dropout", 0.0, 0.8, 0.5)
                with c3:
                    st.markdown("**Training**")
                    lr = st.number_input("Learning Rate", value=0.001, format="%.4f")
                    epochs = st.slider("Epochs", 5, 50, 10)
                
                st.markdown("---")
                
                if st.button("Start Training", use_container_width=True):
                    model = training_utils.build_manual_model(f1, f2, dense, dropout, lr)
                    plot_area = st.empty()
                    fact_box = st.empty()
                    
                    with st.spinner("Training..."):
                        # Show blinking fun facts (no emojis)
                        rotate_fun_facts(fact_box, FUN_FACTS, duration_seconds=2.5, cycles=3)

                        
                        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
                                  callbacks=[training_utils.StreamlitPlotCallback(plot_area)], verbose=0)
                        
                    fact_box.empty()
                    
                    final_acc = history.history['val_accuracy'][-1]
                    log_experiment(
                        method="Manual Tuning", 
                        epochs=epochs, 
                        accuracy=final_acc, 
                        details=f"L1:{f1}, L2:{f2}, Dense:{dense}, DO:{dropout}, LR:{lr}"
                    )
                    
                    if save_and_update_model(model):
                        st.success("Model Saved Successfully!")
                        st.balloons()

        with tab2:
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            
            _, col_center, _ = st.columns([1, 2, 1])
            
            with col_center:
                st.markdown("""
<div class="glass-card" style="margin-bottom: 24px;">
    <div style="text-align: center;">
        <h3 style="color: #1E293B; margin-bottom: 5px;">Auto-Tuning Engine</h3>
        <p style="color: #64748B; font-size: 0.9rem;">Configure search parameters to find the best model.</p>
    </div>
</div>
""", unsafe_allow_html=True)
                
                c_opt1, c_opt2 = st.columns(2)
                with c_opt1:
                    trials = st.slider("Max Trials", 5, 50, 10)
                with c_opt2:
                    epochs_per_trial = st.slider("Epochs / Trial", 5, 50, 5)
                
                st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
                
                if 'tuning_active' not in st.session_state:
                    st.session_state.tuning_active = False
                
                if not st.session_state.tuning_active:
                    if st.button("Start Auto-Tuning", use_container_width=True):
                        st.session_state.tuning_active = True
                        if os.path.exists('my_dir'): shutil.rmtree('my_dir')
                        st.rerun()
            
            if st.session_state.tuning_active:
                
                _, c_stop, _ = st.columns([1, 1, 1])
                with c_stop:
                    if st.button("Stop Tuning & Save Best", use_container_width=True):
                        st.session_state.tuning_active = False
                        st.rerun()

                st.markdown("<hr style='opacity: 0.3; margin: 30px 0;'>", unsafe_allow_html=True)
                
                fact_box = st.empty()
                
                col_graph, col_board = st.columns(2, gap="large")
                
                with col_graph:
                    live_placeholder = st.empty()
                    
                with col_board:
                    leaderboard_placeholder = st.empty()
                
                with live_placeholder.container():
                     st.info("Initializing search space...")
                
                try:
                    # Initial Fact (no emoji)
                    rotate_fun_facts(fact_box, FUN_FACTS, duration_seconds=2.5, cycles=4)

                    # fact_box.markdown(f'<div class="fun-fact-container"><span class="fun-fact-text">{fact}</span></div>', unsafe_allow_html=True)

                    tuner = training_utils.StreamlitTuner(live_placeholder, leaderboard_placeholder, hypermodel=training_utils.build_tuner_model,
                                                        objective='val_accuracy', max_trials=trials,
                                                        executions_per_trial=1, directory='my_dir', project_name='cap_v3')
                    
                    # We can't update fact inside tuner easily without passing it down, keeping it static for now or it will reset on rerun
                    tuner.search(X_train, y_train, epochs=epochs_per_trial, validation_data=(X_test, y_test), verbose=0)
                    
                    # Normal completion
                    st.session_state.tuning_active = False
                    st.rerun()
                    
                except Exception as e:
                    # Ensure state is cleaned up to avoid stuck tuning loop
                    st.session_state.tuning_active = False
                    st.error(f"Optimization Interrupted: {e}")
                    # don't re-raise, let user interact
                    
            if not st.session_state.tuning_active and os.path.exists('my_dir'):
                try:
                    tuner = training_utils.StreamlitTuner(st.empty(), st.empty(), hypermodel=training_utils.build_tuner_model,
                                                        objective='val_accuracy', max_trials=trials,
                                                        executions_per_trial=1, directory='my_dir', project_name='cap_v3')
                    best_hps = tuner.get_best_hyperparameters()[0]
                    
                    if best_hps:
                        model = tuner.hypermodel.build(best_hps)
                        plot_final = st.empty()
                        
                        history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), verbose=0)
                        final_acc = history.history['val_accuracy'][-1]
                        
                        hp_details = f"Filters1:{best_hps.get('conv_1_filter')}, Filters2:{best_hps.get('conv_2_filter')}, Dense:{best_hps.get('dense_units')}, LR:{best_hps.get('lr'):.4f}"
                        log_experiment(f"Bayesian (Trials: {trials})", epochs_per_trial, final_acc, hp_details)
                        
                        if save_and_update_model(model):
                            st.markdown("<br>", unsafe_allow_html=True)
                            _, c_res, _ = st.columns([1, 2, 1])
                            with c_res:
                                st.markdown(f"""
<div class="glass-card" style="background-color: rgba(220, 252, 231, 0.6); border: 1px solid #86EFAC;">
    <h4 style="color: #166534; text-align: center; margin: 0;">Optimization Complete & Model Saved</h4>
    <p style="text-align: center; margin-top: 5px; color: #14532d;">Accuracy: {final_acc:.4f}</p>
</div>
""", unsafe_allow_html=True)
                            st.balloons()
                            shutil.rmtree('my_dir')
                except Exception:
                    # If something goes wrong reading tuner outputs, don't crash the app UI
                    pass
    
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Experiment History</p>', unsafe_allow_html=True)
    
    if os.path.exists("training_history.csv"):
        df = pd.read_csv("training_history.csv")
        df = df.sort_values(by="Timestamp", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No experiments logged yet.")
