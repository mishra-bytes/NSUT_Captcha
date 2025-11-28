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

import backend
import training_utils

st.set_page_config(
    page_title="CAPTCHArd",
    page_icon="assets/logo_white.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

    :root {
        --primary: #2563EB;
        --primary-hover: #1D4ED8;
        --text-main: #1E293B;
        --text-sub: #64748B;
        --glass-bg: rgba(255, 255, 255, 0.75);
        --glass-border: 1px solid rgba(255, 255, 255, 0.5);
        --glass-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
    }

    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        background-attachment: fixed;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: var(--glass-border);
        border-radius: 16px;
        padding: 24px;
        box-shadow: var(--glass-shadow);
        margin-bottom: 24px;
        height: 100%;
    }

    .section-title {
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-sub);
        font-weight: 600;
        margin-bottom: 1rem;
    }

    div.stButton > button {
        background-color: var(--primary);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        width: 100%;
    }
    
    div.stButton > button:hover {
        background-color: var(--primary-hover);
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }

    .prediction-display {
        background: white;
        border: 1px solid #E2E8F0;
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

    img {
        border-radius: 8px;
    }
    
    div[data-baseweb="input"] {
        background-color: white;
        border-radius: 6px;
    }

</style>
""", unsafe_allow_html=True)

def render_custom_loader():
    file_path = "assets/loading.gif"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data_url = base64.b64encode(f.read()).decode("utf-8")
        return st.markdown(
            f'<div style="display: flex; justify-content: center; padding: 2rem;"><img src="data:image/gif;base64,{data_url}" width="40"></div>',
            unsafe_allow_html=True,
        )
    return st.spinner("Processing...")

def get_image_html(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64}"

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
    st.caption("v3.0 Final Version")
    st.caption("Made with &#9829; by Aditya Mishra")

if 'model' not in st.session_state or st.session_state.model is None:
    st.session_state.model = backend.load_pretrained_model()

if 'dataset_uploaded' not in st.session_state:
    st.session_state.dataset_uploaded = False

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
    except Exception as e:
        st.error(f"Save failed: {e}")

if mode == "Live Dashboard":
    
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.5rem;">Live CAPTCHA Inference Dashboard</h1>
        <p style="color: #64748B;">This Computer Vision project fetches a CAPTCHA from the official site of NSUT UMS , and then applies an original segmentation technique and tries to identify the correct sequence using purely Convolution based Neural Networks .</p>    
    </div>
    """, unsafe_allow_html=True)

    col_status, col_action = st.columns([3, 1], gap="medium", vertical_alignment="center")
    
    with col_status:
        status_content = ""
        if st.session_state.model:
            status_content = '<span style="color: #166534; font-weight: 500;">● Online</span>: Model loaded and ready for inference.'
        else:
            status_content = '<span style="color: #DC2626; font-weight: 500;">● Offline</span>: No model detected.'
            
        st.markdown(f"""
        <div class="glass-card" style="padding: 20px; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span class="section-title" style="margin-right: 15px;">System Status</span>
                {status_content}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_action:
        if st.button("Fetch Captcha", use_container_width=True):
            st.session_state.trigger_fetch = True
        else:
            st.session_state.trigger_fetch = False

    if st.session_state.get('trigger_fetch'):
        
        loader_ph = st.empty()
        with loader_ph:
            render_custom_loader()
            
        fetcher = backend.CaptchaFetcher()
        img_bytes, error = fetcher.fetch_single_image()
        time.sleep(0.8)
        loader_ph.empty()

        if error:
            st.error(f"Connection Error: {error}")
        else:
            nparr = np.frombuffer(img_bytes, np.uint8)
            original_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            _, cleaned = backend.preprocess_captcha_v2(io.BytesIO(img_bytes))
            
            col_visuals, col_prediction = st.columns([2, 1], gap="large")
            
            with col_visuals:
                digits = backend.segment_characters_robust(cleaned)
                
                src_html = get_image_html(original_img)
                bin_html = get_image_html(cleaned)
                
                digits_divs = ""
                if len(digits) == 5:
                    for d in digits:
                        digits_divs += f'<div style="flex: 1; text-align: center;"><img src="{get_image_html(d)}" style="width: 100%; border-radius: 4px; border: 1px solid #E2E8F0;"></div>'
                
                st.markdown(f"""
                <div class="glass-card">
                    <p class="section-title">Visual Analysis</p>
                    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                        <div style="flex: 1;">
                            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 5px; font-weight: 500;">Raw Input</div>
                            <img src="{src_html}" style="width: 100%; border-radius: 8px; border: 1px solid #E2E8F0;">
                        </div>
                        <div style="flex: 1;">
                            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 5px; font-weight: 500;">Binary Mask</div>
                            <img src="{bin_html}" style="width: 100%; border-radius: 8px; border: 1px solid #E2E8F0;">
                        </div>
                    </div>
                    <div style="border-top: 1px solid #E2E8F0; margin: 15px 0;"></div>
                    <p class="section-title" style="margin-bottom: 10px;">Segmentation Stream</p>
                    <div style="display: flex; gap: 10px;">
                        {digits_divs}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_prediction:
                if len(digits) == 5 and st.session_state.model:
                    prediction = backend.predict_sequence(st.session_state.model, digits)
                    st.markdown(f"""
                    <div class="glass-card">
                        <p class="section-title">Inference</p>
                        <div class="prediction-display">
                            <div class="prediction-text">{prediction}</div>
                            <div class="confidence-tag">High Confidence</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Segmentation Failed")

elif mode == "Training Studio":
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.5rem;">Training Studio</h1>
        <p style="color: #64748B;">Design, train, and optimize CNN architectures.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 20px;">
        <p class="section-title" style="margin-bottom: 10px;">Data Ingestion</p>
        <p style="font-size: 0.9rem; color: #64748B;">Upload your labeled dataset to begin.</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_stats = st.columns([2, 1])
    
    with col_upload:
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
                    lr = st.number_input("Learning Rate", value=0.0001, format="%.10f")
                    epochs = st.slider("Epochs", 5, 50, 10)
                
                st.markdown("---")
                
                if st.button("Start Training", use_container_width=True):
                    model = training_utils.build_manual_model(f1, f2, dense, dropout, lr)
                    plot_area = st.empty()
                    with st.spinner("Training..."):
                        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
                                  callbacks=[training_utils.StreamlitPlotCallback(plot_area)], verbose=0)
                    save_and_update_model(model)

        with tab2:
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            col_info, col_run = st.columns([2, 1])
            with col_info:
                st.info("Bayesian Optimization automatically finds the best architecture.")
                trials = st.slider("Max Trials", 5, 50, 10)
            
            with col_run:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Start Auto-Tuning", use_container_width=True):
                    if os.path.exists('my_dir'): shutil.rmtree('my_dir')
                    col_stat, col_chart = st.columns([1, 2])
                    with col_stat: st_stat = st.container()
                    with col_chart: st_metric = st.container()
                    
                    tuner = training_utils.StreamlitTuner(st_stat, st_metric, hypermodel=training_utils.build_tuner_model,
                                                        objective='val_accuracy', max_trials=trials,
                                                        executions_per_trial=1, directory='my_dir', project_name='cap_v3')
                    tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=0)
                    
                    best_hps = tuner.get_best_hyperparameters()[0]
                    st.success("Optimization Complete")
                    
                    model = tuner.hypermodel.build(best_hps)
                    plot_final = st.empty()
                    model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test),
                              callbacks=[training_utils.StreamlitPlotCallback(plot_final)], verbose=0)
                    save_and_update_model(model)
