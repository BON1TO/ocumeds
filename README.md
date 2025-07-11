# 🧠 OcuMedAI - FastAPI Medical Predictor API

OcuMedAI is a machine learning-powered FastAPI backend that analyzes a patient's retinal image and clinical data to predict:

- **Diabetic Retinopathy (DR)**
- **Hypertension (HTN) Risk**
- **Estimated HbA1c (Blood Sugar)**
- **Atherosclerosis Risk**

This API is intended for integration with frontend apps or platforms that help screen chronic health conditions using AI.

---

## 🚀 Features

- 🔍 DR detection using CNN model (`DR_predictor.h5`)
- ⚡ Hypertension prediction with regression model
- 🧪 HbA1c estimation via XGBoost
- 💓 Atherosclerosis risk score calculation
- 📦 FastAPI-powered backend
- 🧬 Supports image and form data upload
