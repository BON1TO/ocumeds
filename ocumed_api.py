import os
import subprocess
import logging

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import tensorflow as tf
import joblib
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocumed_api")

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8111",
        "https://ocumedai.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENV flag to skip models (useful for debugging)
SKIP_MODELS = os.getenv("SKIP_MODELS", "false").lower() == "true"

# Global model variables
model_dr = None
model_htn = None
model_hba1c = None
scaler = None

@app.on_event("startup")
async def load_models():
    global model_dr, model_htn, model_hba1c, scaler

    if SKIP_MODELS:
        logger.warning("🟡 SKIP_MODELS is True — skipping model loading.")
        return

    if not os.path.exists("predictors/DR_predictor.h5"):
        logger.info("📦 Model files not found. Downloading...")
        try:
            subprocess.run(["python", "download_models.py"], check=True)
            logger.info("✅ Model files downloaded.")
        except subprocess.CalledProcessError:
            logger.error("❌ Model download failed.")
            raise RuntimeError("Model download failed.")

    try:
        logger.info("🔁 Loading models into memory...")
        model_dr = tf.keras.models.load_model("predictors/DR_predictor.h5")
        model_htn = tf.keras.models.load_model("predictors/HTN_InceptionV3_regression.h5")
        model_hba1c = joblib.load("predictors/hba1c_xgboost_predictor.pkl")
        scaler = joblib.load("predictors/hba1c_scaler.pkl")
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        logger.exception("❌ Failed to load models.")
        raise

@app.get("/")
def home():
    return {"message": "OcuMedAI FastAPI is online"}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    BMI: float = Form(...),
    smokingStatus: str = Form(...)
):
    global model_dr, model_htn, model_hba1c, scaler

    if any(m is None for m in [model_dr, model_htn, model_hba1c, scaler]):
        raise RuntimeError("❌ Models not loaded. Cannot proceed with prediction.")

    def preprocess_image(image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        return np.expand_dims(img_array, axis=0)

    img_data = await image.read()
    input_img = preprocess_image(img_data)

    # DR Prediction
    dr_pred = model_dr.predict(input_img)
    dr_level = np.argmax(dr_pred)
    dr_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']
    dr_result = dr_labels[dr_level]

    # HTN Prediction
    htn_pred = model_htn.predict(input_img)
    htn_percent = float(np.clip(htn_pred[0][0] * 100, 0, 100))
    htn_binary = 1 if htn_percent >= 50 else 0

    # HbA1c Prediction
    sex_val = 1 if sex == "Male" else 0
    smoke_val = 1 if smokingStatus == "Yes" else 0
    diabetes_est = 1 if dr_level > 0 or htn_percent > 60 else 0

    features = np.array([[age, sex_val, BMI, htn_binary, diabetes_est, smoke_val]])
    features_scaled = scaler.transform(features)
    hba1c = float(model_hba1c.predict(features_scaled)[0])

    # Atherosclerosis Risk Calculation
    htn_scaled = htn_percent / 100
    dr_scaled = dr_level / 4
    hba1c_scaled = hba1c / 10
    athero_risk = (
        0.4 * htn_scaled +
        0.3 * dr_scaled +
        0.4 * hba1c_scaled
    ) * 100

    return {
        "diabeticRetinopathyLevel": dr_result,
        "hypertensionRisk": round(htn_percent, 2),
        "hba1cLevel": round(hba1c, 2),
        "atherosclerosisRisk": round(athero_risk, 2)
    }
