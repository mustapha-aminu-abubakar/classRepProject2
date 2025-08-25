import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import efficientnet_b0
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


# --- Config ---
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = 'C:/Users/HP ELITEBOOK/Documents/backend/efficientnet_chicken_disease_updated.pth'
CLASS_NAMES = ["Healthy", "Coccidiosis", "Newcastle Disease (NCD)", "Salmonella"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "jfif"}

# --- Flask app ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Model globals ---
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Configure SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///inference_history.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Model for inference results
class Inference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    label = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
def save_inference(filename, label, confidence):
    record = Inference(filename=filename, label=label, confidence=confidence)
    db.session.add(record)
    db.session.commit()

# --- Helpers ---
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(model_path, num_classes):
    # Load base EfficientNet-B0
    model = efficientnet_b0(weights=None)  # no pretrained weights since checkpoint has everything
    num_ftrs = model.classifier[1].in_features

    # 🔥 Match EXACT training classifier head
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    # Load trained weights
    state = torch.load(model_path, map_location=_device)
    model.load_state_dict(state)  # strict=True will now succeed
    model.to(_device)
    model.eval()
    return model


def bootstrap():
    global _model
    if _model is None:  # load only once
        print("🔄 Loading model...")
        _model = load_model(MODEL_PATH, len(CLASS_NAMES))
        print("✅ Model loaded and ready!")


def infer_image(pil_img: Image.Image):
    if _model is None:
        raise RuntimeError("Model not initialized. Call bootstrap() first.")

    tensor = _transform(pil_img.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    return CLASS_NAMES[pred_idx.item()], conf.item()


# --- Routes ---
@app.route("/")
def index():
    return redirect(url_for("upload"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    global _model
    if _model == None:
        bootstrap()
    if request.method == "POST":
        print(request.files)
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            pil_img = Image.open(save_path)
            label, confidence = infer_image(pil_img)
            
            # Save to DB
            save_inference(filename, label, confidence)
            
            return render_template("result.html",
                                   filename=filename,
                                   label=label,
                                   confidence=confidence)
    return render_template("upload.html")


@app.route("/history")
def history():
    records = Inference.query.order_by(Inference.timestamp.desc()).all()
    return render_template("history.html", records=records)



if __name__ == "__main__":
    bootstrap()  # load the model at startup
    app.run(debug=True)

