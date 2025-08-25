# app.py
import os
import io
import uuid
import sqlite3
from datetime import datetime

from flask import (
    Flask, request, jsonify, g, render_template, redirect, url_for, flash
)
from werkzeug.utils import secure_filename

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch import nn

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "efficientnet_chicken.pth")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
DB_PATH = os.environ.get("DB_PATH", "inferences.db")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff"}

CLASS_NAMES = [
    "Healthy",
    "Coccidiosis", 
    "Newcastle Disease (NCD)",
    "Salmonella",
]

app = Flask(__name__)
app.secret_key = "supersecret"  # required for flash messages
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# Database helpers
# ----------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    db.execute(
        """CREATE TABLE IF NOT EXISTS inferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL
        )"""
    )
    db.commit()

def insert_inference(timestamp, filename, prediction, confidence):
    db = get_db()
    db.execute(
        "INSERT INTO inferences (timestamp, filename, prediction, confidence) VALUES (?, ?, ?, ?)",
        (timestamp, filename, prediction, confidence),
    )
    db.commit()

def fetch_all_inferences():
    db = get_db()
    cur = db.execute("SELECT * FROM inferences ORDER BY id DESC")
    return [dict(row) for row in cur.fetchall()]

# ----------------------------
# Model loading
# ----------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_transform = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def load_model(model_path, num_classes):
    model = build_model(num_classes)
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state)
    model.to(_device)
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def infer_image(pil_img):
    with torch.no_grad():
        tensor = _transform(pil_img.convert("RGB")).unsqueeze(0).to(_device)
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        conf, idx = torch.max(probs, dim=0)
        pred = CLASS_NAMES[int(idx)]
        return pred, float(conf)

# ----------------------------
# App bootstrap
# ----------------------------
@app.before_first_request
def bootstrap():
    init_db()
    global _model, _transform
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = load_model(MODEL_PATH, len(CLASS_NAMES))
    if _transform is None:
        _transform = get_transform()

# ----------------------------
# Routes (HTML + JSON)
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part.")
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Unsupported file type.")
            return redirect(request.url)

        try:
            from io import BytesIO
            image_bytes = file.read()
            pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            flash("Invalid image.")
            return redirect(request.url)

        # Save file
        fname = secure_filename(file.filename)
        unique_name = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}{os.path.splitext(fname)[1].lower()}"
        path = os.path.join(UPLOAD_DIR, unique_name)
        pil_img.save(path)

        # Inference
        pred, conf = infer_image(pil_img)

        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        insert_inference(ts, unique_name, pred, conf)

        flash(f"Prediction: {pred} (Confidence {conf:.2f})")
        return redirect(url_for("history"))

    return render_template("upload.html")

@app.route("/history")
def history():
    records = fetch_all_inferences()
    return render_template("history.html", records=records)

# JSON APIs still available
@app.route("/api/history")
def api_history():
    return jsonify(fetch_all_inferences())

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Bad file type"}), 400
    from io import BytesIO
    pil_img = Image.open(BytesIO(file.read())).convert("RGB")
    fname = secure_filename(file.filename)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}{os.path.splitext(fname)[1].lower()}"
    path = os.path.join(UPLOAD_DIR, unique_name)
    pil_img.save(path)
    pred, conf = infer_image(pil_img)
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    insert_inference(ts, unique_name, pred, conf)
    return jsonify({"filename": unique_name, "prediction": pred, "confidence": conf, "timestamp": ts})

if __name__ == "__main__":
    with app.app_context():
        bootstrap()
    app.run(debug=True)
