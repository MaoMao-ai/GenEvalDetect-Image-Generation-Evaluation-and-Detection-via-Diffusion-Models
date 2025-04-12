import os
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# === Paths (point to folder, not specific files) ===
model_path = r"D:\VS Projects\ECE-580\Proj\Detection"
processor_path = r"D:\VS Projects\ECE-580\Proj\Detection"

# === Load model and processor ===
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(processor_path)

# === Define transforms (must match training) ===
transform = Compose([
    Resize(processor.size["height"]),
    CenterCrop(processor.size["height"]),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std),
])

# === Label mapping ===
id2label = model.config.id2label  # Should be {0: 'adm', 1: 'real'}
model.eval()

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = transform(image).unsqueeze(0)  # shape: (1, 3, H, W)
    with torch.no_grad():
        outputs = model(pixel_values=inputs)
        predicted_label = outputs.logits.argmax(-1).item()
    return id2label.get(predicted_label, "Unknown")

def predict_folder(folder_path):
    print(f"--- Predictions for: {folder_path} ---")
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            label = predict_image(img_path)
            print(f"{filename}: {label}")

# === Run prediction ===
predict_folder(r"D:\VS Projects\ECE-580\Proj\Detection\fake")
predict_folder(r"D:\VS Projects\ECE-580\Proj\Detection\real")
