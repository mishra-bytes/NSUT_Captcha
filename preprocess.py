import cv2
import numpy as np
import os

INPUT_DIR = "Captcha -20250613T230133Z-1-001\Captcha"
OUTPUT_DIR = "Captcha -20250613T230133Z-1-001\Preprocessed"
TARGET_SIZE = (160, 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(INPUT_DIR, file)

        # Load the image in color
        img = cv2.imread(path)

        # Convert to HSV to isolate dark pixels more robustly
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define black color range (tweak if needed)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])  # Vary the V to match your text darkness

        # Create a mask for black pixels
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Resize and pad the mask like before
        h, w = mask.shape
        scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
        resized = cv2.resize(mask, (int(w * scale), int(h * scale)))

        padded = np.ones((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8) * 255
        x_offset = (TARGET_SIZE[0] - resized.shape[1]) // 2
        y_offset = (TARGET_SIZE[1] - resized.shape[0]) // 2
        padded[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

        # Save the result
        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(file)[0] + ".png")
        cv2.imwrite(out_path, padded)

print("✅ Extracted black elements from captchas and saved them to 'black_text_only/'")
