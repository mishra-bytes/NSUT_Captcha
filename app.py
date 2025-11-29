import streamlit as st
import numpy as np
import cv2
import os
import zipfile
import io
import shutil
import time
import base64
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Import our custom modules
import backend
import training_utils

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
        --glass-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
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
        margin-bottom: 24px; /* Ensure spacing between elements */
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

    /* BUTTONS */
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
        font-size: 3rem;
        font-weight: 700;
        color: var(--primary);
        letter-spacing: 0.5rem;
        line-height: 1;
        margin: 10px 0;
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

</style>
""", unsafe_allow_html=True)

# --- HELPER: CUSTOM LOADER ---
def render_custom_loader():
    file_path = "assets/loading.gif"
    if os.path.exists(file_path):
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

# --- INIT SESSION ---
if 'model' not in st.session_state or st.session_state.model is None:
    st.session_state.model = backend.load_pretrained_model()

if 'dataset_uploaded' not in st.session_state:
    st.session_state.dataset_uploaded = False

# --- UTILS ---
def load_uploaded_dataset(uploaded_file):
    with zipfile.ZipFile(uploaded_file, 'r') as z:
        z.extractall("temp_dataset")
    data, labels = [], []
    valid_count = 0
    for root, dirs, files in os.walk("temp_dataset"):
        for f in files:
            if f.endswith(('.png', '.jpg')):
                label_text = os.path.splitext(f)[0]
                if len(label_text) != 5 or not label_text.isdigit(): continue
                path = os.path.join(root, f)
                _, cleaned = backend.preprocess_captcha_v2(path)
                digits = backend.segment_characters_robust(cleaned)
                if len(digits) == 5:
                    valid_count += 1
                    for i, d in enumerate(digits):
                        img = d / 255.0
                        img = np.expand_dims(img, axis=-1)
                        data.append(img)
                        labels.append(int(label_text[i]))
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

    # --- LAYOUT ---
    col_main, col_sidebar = st.columns([7, 3], gap="large")
    
    # === LEFT COLUMN ===
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

    # === RIGHT COLUMN ===
    with col_sidebar:
        st.markdown('<div style="height: 0px;"></div>', unsafe_allow_html=True)
        if st.button("Fetch Captcha", use_container_width=True):
            st.session_state.trigger_fetch = True
        else:
            pass 

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
                <div class="glass-card">
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
             st.markdown("""
             <div class="glass-card" style="height: 300px; display: flex; align-items: center; justify-content: center; color: #94A3B8;">
                <div>Waiting for input...</div>
             </div>
             """, unsafe_allow_html=True)


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
    
    # 1. DATASET
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
            with st.spinner("Processing..."):
                X, y, count = load_uploaded_dataset(upload)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.dataset_uploaded = True
                st.session_state.sample_count = count
                
        if st.session_state.dataset_uploaded:
             st.success(f"Ready: {st.session_state.sample_count} Samples")
        else:
             st.info("Waiting for upload...")

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
                    with st.spinner("Training..."):
                        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
                                  callbacks=[training_utils.StreamlitPlotCallback(plot_area)], verbose=0)
                    if save_and_update_model(model):
                        st.success("✅ Model Saved Successfully!")
                        st.balloons()

        with tab2:
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            
            # --- CENTRALLY ALIGNED OPTIMIZATION PANEL ---
            # Using 3 columns [1, 2, 1] to center the middle content
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
                    trials = st.slider("Max Trials", 5, 50, 10, help="Total number of different models to test.")
                with c_opt2:
                    epochs_per_trial = st.slider("Epochs / Trial", 5, 50, 5, help="How long to train each model.")
                
                st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
                
                if st.button("🚀 Start Auto-Tuning", use_container_width=True):
                    if os.path.exists('my_dir'): shutil.rmtree('my_dir')
                    
                    # Create two placeholders for the layout
                    # We use st.empty() so we can clear them if needed
                    live_placeholder = st.empty()
                    leaderboard_placeholder = st.empty()
                    
                    # Initialize the layout with columns inside the placeholders
                    with live_placeholder.container():
                         st.info("Initializing search space...")
                    
                    tuner = training_utils.StreamlitTuner(live_placeholder, leaderboard_placeholder, hypermodel=training_utils.build_tuner_model,
                                                        objective='val_accuracy', max_trials=trials,
                                                        executions_per_trial=1, directory='my_dir', project_name='cap_v3')
                    
                    tuner.search(X_train, y_train, epochs=epochs_per_trial, validation_data=(X_test, y_test), verbose=0)
                    
                    best_hps = tuner.get_best_hyperparameters()[0]
                    
                    model = tuner.hypermodel.build(best_hps)
                    plot_final = st.empty()
                    model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test),
                              callbacks=[training_utils.StreamlitPlotCallback(plot_final)], verbose=0)
                    
                    if save_and_update_model(model):
                        st.markdown("""
                        <div class="glass-card" style="background-color: #DCFCE7; border: 1px solid #86EFAC;">
                            <h4 style="color: #166534; text-align: center; margin: 0;">🎉 Optimization Complete & Model Saved!</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()