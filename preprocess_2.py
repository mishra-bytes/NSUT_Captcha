import cv2
import numpy as np
from pathlib import Path

# Input and output directories
input_dir = Path("Preprocessed")
output_dir = Path("Processed_Techniques")

# Basic image processing functions
def gaussian_blur(img): return cv2.GaussianBlur(img, (5, 5), 0)
def median_blur(img): return cv2.medianBlur(img, 5)
def bilateral_filter(img): return cv2.bilateralFilter(img, 9, 75, 75)
def canny_edge(img): return cv2.Canny(img, 50, 150)
def sobel_x(img): return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
def sobel_y(img): return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
def sobel_magnitude(img):
    x = sobel_x(img)
    y = sobel_y(img)
    return cv2.magnitude(x, y)

# Normalize float64 gradients to uint8
def normalize_sobel(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Define techniques and combinations
techniques = {
    "gaussian_blur": gaussian_blur,
    "median_blur": median_blur,
    "bilateral_filter": bilateral_filter,
    "canny_edge": canny_edge,
    "sobel_x": sobel_x,
    "sobel_y": sobel_y,
    "sobel_magnitude": sobel_magnitude,
    # Combined operations
    "gaussian_then_canny": lambda img: canny_edge(gaussian_blur(img)),
    "median_then_canny": lambda img: canny_edge(median_blur(img)),
    "bilateral_then_canny": lambda img: canny_edge(bilateral_filter(img)),
    "gaussian_then_sobel_mag": lambda img: sobel_magnitude(gaussian_blur(img)),
    "median_then_sobel_mag": lambda img: sobel_magnitude(median_blur(img)),
    "bilateral_then_sobel_mag": lambda img: sobel_magnitude(bilateral_filter(img)),
}

# Create output directories
for name in techniques:
    (output_dir / name).mkdir(parents=True, exist_ok=True)

# Process images
for img_path in input_dir.glob("*.png"):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    for tech_name, func in techniques.items():
        processed = func(img)

        # Normalize if it's a Sobel or magnitude result
        if "sobel" in tech_name and processed.dtype != np.uint8:
            processed = normalize_sobel(processed)

        save_path = output_dir / tech_name / img_path.name
        cv2.imwrite(str(save_path), processed)

print("✅ Batch processing complete. Check the 'Processed_Techniques/' folder.")
