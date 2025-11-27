import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import random
import time
import os
import tensorflow as tf
import streamlit as st

# --- VISION PIPELINE (The Golden Logic) ---

def preprocess_captcha_v2(image_path):
    # Read image
    if isinstance(image_path, str):
        img = cv2.imread(image_path, 0)
    else:
        # Handle byte stream from upload/fetch
        nparr = np.frombuffer(image_path.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None: return None, None
    
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return img, cleaned

def segment_characters_robust(cleaned_img):
    contours, _ = cv2.findContours(cleaned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 10: 
            boundingBoxes.append((x, y, w, h))
    boundingBoxes.sort(key=lambda x: x[0])
    
    digit_images = []
    if len(boundingBoxes) == 0: return []
    
    total_width = sum([w for (x, y, w, h) in boundingBoxes])
    avg_width = total_width / 5
    
    for (x, y, w, h) in boundingBoxes:
        num_digits = int(round(w / avg_width))
        if num_digits == 0: num_digits = 1
        step_w = w // num_digits
        
        for i in range(num_digits):
            new_x = x + (i * step_w)
            new_w = step_w
            roi = cleaned_img[y:y+h, new_x:new_x+new_w]
            
            target_size = 32
            h_roi, w_roi = roi.shape
            final_digit = np.zeros((target_size, target_size), dtype=np.uint8)
            scale = min(28/h_roi, 28/w_roi)
            n_w = int(w_roi * scale)
            n_h = int(h_roi * scale)
            
            if n_w > 0 and n_h > 0:
                rsz = cv2.resize(roi, (n_w, n_h))
                start_x = (target_size - n_w) // 2
                start_y = (target_size - n_h) // 2
                final_digit[start_y:start_y+n_h, start_x:start_x+n_w] = rsz
                digit_images.append(final_digit)
    return digit_images

# --- CAPTCHA FETCHER (Refactored for Web App) ---

class CaptchaFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
        ]
        self._configure_session()

    def _configure_session(self):
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://imsnsit.org/imsnsit/',
            'User-Agent': random.choice(self.user_agents)
        }
        self.session.headers.update(headers)

    def fetch_single_image(self):
        """Fetches a single captcha and returns the binary content."""
        try:
            # 1. Init Session
            self.session.get('https://imsnsit.org/imsnsit/')
            self.session.get('https://www.imsnsit.org/imsnsit/student_login110.php')
            
            # 2. Go to Login Page
            login_url = 'https://www.imsnsit.org/imsnsit/student_login.php'
            response = self.session.get(login_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            captcha_img = soup.find('img', {'id': 'captchaimg'})
            if not captcha_img:
                return None, "Captcha element not found."

            # 3. Download Image
            captcha_url = urljoin(login_url, captcha_img['src'])
            img_response = self.session.get(captcha_url, headers={'Referer': login_url})
            
            return img_response.content, None
            
        except Exception as e:
            return None, str(e)

# --- MODEL UTILS ---

def predict_sequence(model, digit_images):
    if len(digit_images) != 5:
        return "Error: Segmentation Failed"
    
    # Batch processing
    batch = np.array(digit_images) / 255.0
    batch = np.expand_dims(batch, axis=-1)
    
    preds = model.predict(batch, verbose=0)
    pred_indices = np.argmax(preds, axis=1)
    
    return "".join(map(str, pred_indices))

@st.cache_resource
def load_pretrained_model():
    """
    Loads the model efficiently. 
    @st.cache_resource ensures the model is loaded only ONCE per server restart,
    not per user session.
    """
    model_path = './model/final_captcha_model.h5'
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None