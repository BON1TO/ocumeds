# 🧠 OcuMeds – Diabetic Retinopathy Detection API

OcuMeds is an AI-powered backend API that helps detect **diabetic retinopathy**, a serious eye condition caused by diabetes, using retinal fundus images. It uses deep learning to analyze images and return the most likely diagnosis within seconds.

This project aims to support early detection and screening efforts through an accessible and scalable REST API.

---

## 🔍 What It Does

- Accepts eye images through a simple API endpoint
- Uses a trained Convolutional Neural Network (CNN) to detect signs of diabetic retinopathy
- Returns the predicted stage of the disease (e.g., No DR, Mild, Moderate, Severe)
- Designed for integration with web apps, diagnostic platforms, or mobile health solutions

---

## 🚀 How It Works

1. The user uploads a retina image via the API
2. The image is preprocessed using OpenCV
3. The trained deep learning model (built with TensorFlow) makes a prediction
4. The API returns the result in JSON format, including prediction confidence

---

## 🛠 Technologies Used

- **FastAPI** – Modern Python framework for building APIs
- **TensorFlow / Keras** – For the deep learning model
- **OpenCV** – For image preprocessing
- **Docker** – For containerization
- **Azure App Service** – Cloud hosting platform
- **Azure Container Registry** – Secure storage for Docker images

---

## ☁️ Deployment

OcuMeds is fully containerized using Docker and deployed on **Microsoft Azure**. This ensures a reliable, scalable, and fast response API, ready to integrate into any application needing diabetic retinopathy detection.

---

