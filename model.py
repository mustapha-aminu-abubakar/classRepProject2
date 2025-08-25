import os
from datetime import datetime
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    # Try common locations
    possible_paths = [
        "models/efficientnet_chicken_disease_updated.pth",
        "./efficientnet_chicken_disease_updated.pth",
        r"C:\Users\alameen umar\Desktop\final year project\efficientnet_chicken_disease_updated.pth"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            break

def load_model():
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}. Using fallback mode.")
        return None
    
    try:
        model = efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 4)
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}. Using fallback mode.")
        return None

# Load model at startup
model = load_model()

CLASS_NAMES = [
    "Healthy",
    "Coccidiosis", 
    "Newcastle Disease (NCD)",
    "Salmonella",
]

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def save_image(file):
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)
    
    # Reset file position for potential reuse
    await file.seek(0)
    return filepath, filename

async def detect_with_model(file):
    """Use the actual ML model for detection"""
    if model is None:
        return None
        
    filepath, filename = await save_image(file)
    image = Image.open(filepath).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        
        result = {
            "filename": filename,
            "image_url": f"/uploads/{filename}",
            "prediction": CLASS_NAMES[pred.item()],
            "confidence": float(conf.item()),
        }
    return result
