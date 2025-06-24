from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

# Load processor and model with tokenizer fix
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten", use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

# Use CPU if no GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load your CAPTCHA image
image_path = r"D:\Work\Projects\Project_NSUT_Captcha\NSUT_Captcha\Processed_Techniques\bilateral_then_canny\captcha_1750587870_0.png"
image = Image.open(image_path).convert("RGB")

# Preprocess and inference
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values)
raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Filter: only keep digits, trim to 5
captcha = re.sub(r'\D', '', raw_text)[:5].ljust(5, '_')

print("Raw OCR Output    :", raw_text)
print("Predicted CAPTCHA :", captcha)
