import streamlit as st
import numpy as np
import cv2
import os
import zipfile
import io
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt

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
    # Check if logo exists to avoid errors if assets missing
    if os.path.exists("./assets/logo_white.png"):
        st.image("./assets/logo_white.png", width=300)
    else:
        st.title("CAPTCHArd")
        
    st.markdown("---")
    mode = st.radio("Select Mode", ["Live Inference", "Training Studio"])
    st.markdown("---")
    st.info("A Computer Vision project demonstrating robust segmentation and CNN classification on complex captchas.")
    st.markdown("---")
    st.caption("Developed with TensorFlow & OpenCV")

# --- SESSION STATE INITIALIZATION ---
# Try to load the model immediately on startup using the cached function
if 'model' not in st.session_state or st.session_state.model is None:
    st.session_state.model = backend.load_pretrained_model()

if 'dataset_uploaded' not in st.session_state:
    st.session_state.dataset_uploaded = False

# --- FUNCTION: LOAD DATA ---
def load_uploaded_dataset(uploaded_file):
    with zipfile.ZipFile(uploaded_file, 'r') as z:
        # Extract to temp
        z.extractall("temp_dataset")
        
    data = []
    labels = []
    valid_count = 0
    
    # Walk through the extracted files
    for root, dirs, files in os.walk("temp_dataset"):
        for f in files:
            if f.endswith(('.png', '.jpg')):
                label_text = os.path.splitext(f)[0]
                # Try to clean label if it's not pure digits (sometimes MacOS adds ._ files)
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

#Helper function to save model safely
def save_and_update_model(model):
    """Saves model to the 'model' directory, creating it if needed."""
    try:
        # 1. Create directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        # 2. Construct path
        save_path = os.path.join('model', 'final_captcha_model.h5')
        
        # 3. Save
        model.save(save_path)
        
        # 4. Update session state
        st.session_state.model = model
        st.toast(f"Model saved to {save_path}! It will be preloaded next time.")
        st.success("✅ Model saved successfully.")
    except Exception as e:
        st.error(f"Failed to save model: {e}")

# ==========================================
# MODE 1: LIVE INFERENCE
# ==========================================
if mode == "Live Inference":
    st.header("CAPTCHArd : The Live Captcha Solver")
    st.markdown("Fetch a real-time captcha from the source, segment it, and predict using the loaded model.")
    
    col_status, col_btn = st.columns([3, 1])
    state = True
    with col_status:
        if st.session_state.model is not None and state==True:
            st.success("✅ Model Preloaded")
            state = False 
        elif st.session_state.model is not None and state==False: 
            st.success("✅ Model Loaded and Ready")   
        else:
            st.warning("No Model Found! Upload 'final_captcha_model.h5' to 'model/' folder or train in Studio.")

    with col_btn:
        fetch_btn = st.button("Fetch Live Captcha", use_container_width=True)

    if fetch_btn:
        fetcher = backend.CaptchaFetcher()
        with st.spinner("Connecting to source..."):
            img_bytes, error = fetcher.fetch_single_image()
        
        if error:
            st.error(f"Failed to fetch: {error}")
        else:
            # Process
            nparr = np.frombuffer(img_bytes, np.uint8)
            original_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            _, cleaned = backend.preprocess_captcha_v2(io.BytesIO(img_bytes))
            
            # Display
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img, caption="Original Source", width=250)
            with col2:
                st.image(cleaned, caption="Processed Binary", width=250)
                
            # Segment
            digits = backend.segment_characters_robust(cleaned)
            
            if len(digits) == 5:
                st.subheader("Segmentation")
                cols = st.columns(5)
                for i, d in enumerate(digits):
                    with cols[i]:
                        st.image(d, caption=f"Digit {i+1}")
                
                # Predict
                if st.session_state.model:
                    prediction = backend.predict_sequence(st.session_state.model, digits)
                    st.success(f"## Prediction: {prediction}")
                else:
                    st.error("Model not loaded, cannot predict.")
            else:
                st.error(f"Segmentation failed. Found {len(digits)} digits.")

# ==========================================
# MODE 2: TRAINING STUDIO
# ==========================================
elif mode == "Training Studio":
    st.header("Model Training Studio")
    st.markdown("Design your CNN architecture or use Bayesian Optimization to find the perfect hyperparameters.")
    
    # 1. DATA UPLOAD
    st.subheader("1. Dataset")
    upload = st.file_uploader("Upload labeled Images (ZIP)", type="zip")
    
    if upload:
        if not st.session_state.dataset_uploaded:
            with st.spinner("Processing Dataset (Preprocessing & Segmenting)..."):
                X, y, count = load_uploaded_dataset(upload)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.dataset_uploaded = True
                st.session_state.sample_count = count
                
        st.success(f"Processed {st.session_state.sample_count} captchas ({len(st.session_state.y)} digit samples).")
    
    if st.session_state.dataset_uploaded:
        X = st.session_state.X
        y = st.session_state.y
        y_encoded = to_categorical(y, 10)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # 2. CONFIGURATION
        st.subheader("2. Configuration")
        tab1, tab2 = st.tabs(["Manual Tuning", "Bayesian Auto-Tuning"])
        
        # --- TAB 1: MANUAL ---
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                f1 = st.slider("Conv Layer 1 Filters", 16, 128, 32, step=16)
                lr_slider = st.slider(
                    "Learning Rate (Slider)",
                    min_value=0.00001,
                    max_value=0.01, # Fixed realistic max for slider
                    step=0.00001,
                    value=0.001,
                    format="%.5f"
                )
                lr = lr_slider
            with col2:
                f2 = st.slider("Conv Layer 2 Filters", 32, 128, 64, step=32)
                epochs = st.slider("Epochs", 5, 50, 10)
            with col3:
                dense = st.slider("Dense Units", 32, 512, 64, step=32)
                dropout = st.slider("Dropout", 0.0, 0.8, 0.4)
            
            start_manual = st.button("Start Training (Manual)")
            
            if start_manual:
                model = training_utils.build_manual_model(f1, f2, dense, dropout, lr)
                plot_placeholder = st.empty()
                
                with st.spinner("Training in progress..."):
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[training_utils.StreamlitPlotCallback(plot_placeholder)],
                        verbose=0
                    )
                st.success("Training Complete!")
                save_and_update_model(model)
                
        # --- TAB 2: BAYESIAN ---
        with tab2:
            st.info("Bayesian Optimization uses probability to find the best hyperparameters intelligently.")
            st.info("PS : The more the number of trials , the more you'll have to wait. (About 1 min per trial.)")
            max_trials = st.slider("Max Trials (Model Variations)", 5, 50, 10)
            start_bayes = st.button("Start Auto-Tuning")
            
            if start_bayes:
                tuner = kt.BayesianOptimization(
                    training_utils.build_tuner_model,
                    objective='val_accuracy',
                    max_trials=max_trials,
                    executions_per_trial=1,
                    directory='my_dir',
                    project_name='captcha_tuning_web'
                )
                
                st.write("Searching for best architecture...")
                tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
                
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                st.json(best_hps.values)
                
                st.write("Retraining best model...")
                model = tuner.hypermodel.build(best_hps)
                plot_placeholder_bayes = st.empty()
                
                model.fit(
                    X_train, y_train,
                    epochs=15,
                    validation_data=(X_test, y_test),
                    callbacks=[training_utils.StreamlitPlotCallback(plot_placeholder_bayes)],
                    verbose=0
                )
                st.success("Auto-Tuning Complete!")
                save_and_update_model(model)

    else:
        st.info("Please upload a dataset (ZIP of images) to unlock the Training Studio.")