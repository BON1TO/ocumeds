# OcuMeds – Diabetic Retinopathy Detection API

OcuMeds is a FastAPI-based machine learning application that analyzes eye images to detect diabetic retinopathy using a pre-trained convolutional neural network (CNN).

> 🚀 Deployed via **Docker** and **Microsoft Azure**.

---

## 🔬 Features

- Upload eye images via REST API
- Predict diabetic retinopathy stage using a trained model
- Built with TensorFlow, FastAPI, and OpenCV
- Deployed with Docker container on Azure
- Supports CORS for frontend communication

---

## 🐳 Dockerized Deployment

### 🔧 Build Docker Image
```bash
docker build -t ocumeds-api .
