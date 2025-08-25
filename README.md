# 🐔 Chicken Disease Detection (Flask + PyTorch)

A simple **Flask web application** that uses a fine-tuned **EfficientNet-B0** model to detect chicken diseases from uploaded images.  
All inferences are stored in a **SQLite database** and can be viewed on the `/history` page.

---

## 🚀 Features
- Upload chicken images (`.png`, `.jpg`, `.jpeg`, `.webp`)  
- Perform inference using EfficientNet-B0  
- Save predictions (filename, label, confidence, timestamp) in a database  
- View inference history in a Bootstrap-styled table  

---

## 📦 Requirements
- Python **3.9+** recommended  
- Virtual environment (`venv`)  

---

## 🔧 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/chicken-disease-detection.git
cd chicken-disease-detection
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install flask flask_sqlalchemy torch torchvision pillow
```

### 4. Add your trained model
Place your trained model file in the project root directory:
```
efficientnet_chicken_disease_updated.pth
```

### 5. Initialize the database
Open a Python shell and run:
```bash
python
```
Then inside Python:
```python
from app import db
db.create_all()
exit()
```

This creates the SQLite database file `inference_history.db`.

### 6. Run the Flask app
```bash
flask run
```

The app will be available at:
```
http://127.0.0.1:5000
```

---

## 📂 Project Structure
```
.
├── app.py                                 # Main Flask application
├── templates/
│   ├── upload.html                        # Upload page
│   └── history.html                       # History page
├── uploads/                               # Uploaded images
├── efficientnet_chicken_disease_updated.pth  # Trained model (you provide)
├── inference_history.db                   # SQLite database (auto-created)
└── README.md
```

---

## 🌐 Routes
- `/upload` → Upload an image and get prediction  
- `/history` → View inference history table  

---

## 📝 Notes
- Ensure your **model architecture in `app.py`** matches exactly how it was trained.  
- Update `MODEL_PATH` and `CLASS_NAMES` in `app.py` if you retrain your model.  
- Uploaded images are stored in the `uploads/` folder.  
