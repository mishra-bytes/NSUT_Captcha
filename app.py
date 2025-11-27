import streamlit as st
import numpy as np
import cv2
import os
import zipfile
import io
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Import our custom modules
import backend
import training_utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CAPTCHArd",
    page_icon="./assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists("./assets/logo_white.png"):
        st.image("./assets/logo_white.png", width=300)
    else:
        st.title("CAPTCHArd")
        st.caption("v2.0 Professional Edition")
        
    st.markdown("---")
    mode = st.radio("NAVIGATION", ["Live Inference", "Training Studio"])
    st.markdown("---")
    st.info("A Computer Vision project demonstrating robust segmentation and CNN classification on complex captchas.")

# --- SESSION STATE INITIALIZATION ---
if 'model' not in st.session_state or st.session_state.model is None:
    st.session_state.model = backend.load_pretrained_model()

if 'dataset_uploaded' not in st.session_state:
    st.session_state.dataset_uploaded = False

# --- UTILS ---
def load_uploaded_dataset(uploaded_file):
    with zipfile.ZipFile(uploaded_file, 'r') as z:
        z.extractall("temp_dataset")
        
    data = []
    labels = []
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
        save_path = os.path.join('model', 'final_captcha_model.h5')
        model.save(save_path)
        st.session_state.model = model
        st.toast(f"Model saved to {save_path}!")
        st.success("✅ Model saved successfully.")
    except Exception as e:
        st.error(f"Failed to save model: {e}")

# ==========================================
# MODE 1: LIVE INFERENCE
# ==========================================
if mode == "Live Inference":
    st.markdown("# CAPTCHArd : Live Captcha Solver")
    st.markdown("Fetch a real-time captcha from the source, segment it, and predict.")
    
    # Cleaner Status Card Implementation
    with st.container():
        col_status, col_btn = st.columns([3, 1])
        with col_status:
            if st.session_state.model is not None:
                st.info("✅ **System Ready:** Model Loaded & Active")
            else:
                st.error("**System Error:** No Model Found. Please train a model first.")
        with col_btn:
            fetch_btn = st.button("FETCH NEW CAPTCHA", use_container_width=True)

    st.markdown("---")

    if fetch_btn:
        fetcher = backend.CaptchaFetcher()
        with st.spinner("Connecting to imsnsit.org..."):
            img_bytes, error = fetcher.fetch_single_image()
        
        if error:
            st.error(f"Failed to fetch: {error}")
        else:
            nparr = np.frombuffer(img_bytes, np.uint8)
            original_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            _, cleaned = backend.preprocess_captcha_v2(io.BytesIO(img_bytes))
            
            # Card-like display
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img, caption="Original Source", use_column_width=True)
            with col2:
                st.image(cleaned, caption="Processed Binary", use_column_width=True)
                
            digits = backend.segment_characters_robust(cleaned)
            
            if len(digits) == 5:
                st.markdown("### Segmentation & Prediction")
                cols = st.columns(5)
                for i, d in enumerate(digits):
                    with cols[i]:
                        st.image(d, width=60)
                
                if st.session_state.model:
                    prediction = backend.predict_sequence(st.session_state.model, digits)
                    st.metric(label="Predicted Captcha", value=prediction)
                else:
                    st.error("Model not loaded.")
            else:
                st.warning(f"Segmentation uncertain. Found {len(digits)} segments.")

# ==========================================
# MODE 2: TRAINING STUDIO
# ==========================================
elif mode == "Training Studio":
    st.markdown("## Model Training Studio")
    
    # 1. DATASET CARD
    st.markdown("#### 1. Dataset")
    with st.container():
        upload = st.file_uploader("Upload labeled Images (ZIP)", type="zip")
        
        if upload and not st.session_state.dataset_uploaded:
            with st.spinner("Preprocessing & Segmenting Dataset..."):
                X, y, count = load_uploaded_dataset(upload)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.dataset_uploaded = True
                st.session_state.sample_count = count
        
        if st.session_state.dataset_uploaded:
            st.success(f"✅ Processed {st.session_state.sample_count} captchas ({len(st.session_state.y)} digits).")

    if st.session_state.dataset_uploaded:
        X = st.session_state.X
        y = st.session_state.y
        y_encoded = to_categorical(y, 10)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # 2. CONFIG CARD
        st.markdown("#### 2. Configuration")
        
        tab1, tab2 = st.tabs(["MANUAL TUNING", "BAYESIAN AUTO-TUNING"])
        
        # --- TAB 1: MANUAL ---
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                f1 = st.slider("Conv 1 Filters", 16, 128, 32, step=16)
                lr = st.number_input("Learning Rate", value=0.001, format="%.10f")
            with col2:
                f2 = st.slider("Conv 2 Filters", 32, 128, 64, step=32)
                epochs = st.slider("Epochs", 5, 50, 10)
            with col3:
                dense = st.slider("Dense Units", 32, 512, 64, step=32)
                dropout = st.slider("Dropout", 0.0, 0.8, 0.4)
            
            if st.button("START MANUAL TRAINING", use_container_width=True):
                model = training_utils.build_manual_model(f1, f2, dense, dropout, lr)
                plot_placeholder = st.empty()
                
                with st.spinner("Training..."):
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[training_utils.StreamlitPlotCallback(plot_placeholder)],
                        verbose=0
                    )
                save_and_update_model(model)

        # --- TAB 2: BAYESIAN ---
        with tab2:
            st.info("💡 Bayesian Optimization intelligently finds the best parameters.")
            
            # Updated Slider Range (5 to 50)
            max_trials = st.slider("Number of Trials", min_value=5, max_value=50, value=10)
            
            if st.button("✨ START AUTO-TUNING", use_container_width=True):
                
                # Create Layout for Real-time Feedback
                col_status, col_leaderboard = st.columns([1, 2])
                
                with col_status:
                    st_status = st.container()
                with col_leaderboard:
                    st_metrics = st.container()
                
                # Clean up previous runs
                if os.path.exists('my_dir'):
                    shutil.rmtree('my_dir')
                
                # Instantiate our Custom StreamlitTuner
                tuner = training_utils.StreamlitTuner(
                    st_status_container=st_status,
                    st_metrics_container=st_metrics,
                    hypermodel=training_utils.build_tuner_model,
                    objective='val_accuracy',
                    max_trials=max_trials,
                    executions_per_trial=1,
                    directory='my_dir',
                    project_name='captcha_tuning_v2'
                )
                
                tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=0)
                
                # Finalizing
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                st.success("Optimization Complete!")
                st.markdown("### Best Hyperparameters:")
                st.json(best_hps.values)
                
                st.write("Retraining best model for final deployment...")
                model = tuner.hypermodel.build(best_hps)
                
                # Simple progress for final train
                final_plot = st.empty()
                model.fit(
                    X_train, y_train,
                    epochs=15,
                    validation_data=(X_test, y_test),
                    callbacks=[training_utils.StreamlitPlotCallback(final_plot)],
                    verbose=0
                )
                save_and_update_model(model)
    else:
        st.warning("👆 Waiting for dataset upload...")