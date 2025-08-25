from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from PIL import Image
import io
import base64
import numpy as np
from typing import Dict, Any

from auth import verify_token
from db import init_db, log_detection
from model import detect_with_model, UPLOAD_DIR

app = FastAPI(title="Chicken Disease Detection API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    init_db()
    print("Database initialized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Chicken Disease Detection API is running"}

@app.post("/detect-disease")
async def detect_disease(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    try:
        # Try to use the ML model first
        ml_result = await detect_with_model(file)
        
        if ml_result:
            # Use ML model results
            disease_name = ml_result["prediction"]
            confidence = ml_result["confidence"] * 100  # Convert to percentage
            severity = get_severity_from_confidence(confidence)
            image_url = ml_result["image_url"]
            filename = ml_result["filename"]
        else:
            # Fallback to rule-based approach
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            disease_name, confidence, severity = analyze_image_simple(image)
            
            # Convert image to base64 for storage
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_url = f"data:image/jpeg;base64,{img_str}"
            filename = file.filename

        # Generate medication recommendations
        medication = get_medication_recommendation(disease_name)
        
        log_detection(filename, image_url, 1)
        
        return {
            "disease": disease_name,
            "confidence": confidence,
            "severity": severity,
            "medication": medication,
            "description": get_disease_description(disease_name),
            "image_url": image_url,
            "filename": filename
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )

def get_severity_from_confidence(confidence: float) -> str:
    if confidence >= 80:
        return "high"
    elif confidence >= 60:
        return "medium"
    else:
        return "low"

def analyze_image_simple(image: Image.Image) -> tuple[str, float, str]:
    """Simple rule-based analysis for demo purposes"""
    img_array = np.array(image)
    
    avg_r = np.mean(img_array[:, :, 0])
    avg_g = np.mean(img_array[:, :, 1])
    avg_b = np.mean(img_array[:, :, 2])
    
    if avg_r > 150 and avg_g > 150 and avg_b > 150:
        return "Healthy", 85.0, "low"
    elif avg_r > 120 and avg_g < 100:
        return "Coccidiosis", 75.0, "medium"
    elif avg_b > 120 and avg_r < 100:
        return "Salmonella", 70.0, "medium"
    else:
        return "Healthy", 60.0, "low"

def get_medication_recommendation(disease: str) -> Dict[str, str]:
    """Get medication recommendations based on disease"""
    recommendations = {
        "Healthy": {
            "name": "None needed",
            "dosage": "N/A",
            "duration": "N/A",
            "instructions": "Continue with regular care and monitoring"
        },
        "Coccidiosis": {
            "name": "Amprolium (Corid)",
            "dosage": "0.012% in drinking water",
            "duration": "5-7 days",
            "instructions": "Add to drinking water. Ensure all birds have access."
        },
        "Salmonella": {
            "name": "Enrofloxacin (Baytril)",
            "dosage": "10-15 mg/kg body weight",
            "duration": "5-7 days",
            "instructions": "Administer orally or in drinking water. Consult vet for exact dosage."
        },
        "Newcastle Disease": {
            "name": "Supportive care + Vaccination",
            "dosage": "Vaccine as per manufacturer",
            "duration": "Immediate + ongoing",
            "instructions": "Isolate affected birds. Vaccinate healthy flock. Consult vet immediately."
        }
    }
    return recommendations.get(disease, recommendations["Healthy"])

def get_disease_description(disease: str) -> str:
    """Get disease descriptions"""
    descriptions = {
        "Healthy": "No signs of disease detected. Continue with regular monitoring and care.",
        "Coccidiosis": "A parasitic disease affecting the intestinal tract, causing diarrhea and weight loss.",
        "Salmonella": "A bacterial infection that can cause diarrhea, lethargy, and decreased egg production.",
        "Newcastle Disease": "A highly contagious viral disease affecting respiratory, nervous, and digestive systems."
    }
    return descriptions.get(disease, "Disease description not available.")

if __name__ == "__main__":
    print("Starting Chicken Disease Detection API...")
    print("Note: This is a demo version with rule-based analysis.")
    print("To use the real ML model, update the model path in main.py")
    uvicorn.run(app, host="0.0.0.0", port=8000)
