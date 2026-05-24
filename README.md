# 🫀 CardioAI - Heart Disease Prediction Web Application
<p align="center">
  <img src="https://img.icons8.com/3d-fluency/94/heart-with-pulse.png" alt="HeartSense AI Logo" width="80"/>
</p>
A premium AI-powered heart disease prediction system built with React, Flask, and Scikit-learn.
<h1 align="center">HeartSense AI — Heart Disease Prediction</h1>
![CardioAI](https://img.shields.io/badge/CardioAI-Heart%20Disease%20Prediction-ff4d6d?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=flat-square)
<p align="center">
  <strong>An end-to-end Machine Learning web application that predicts cardiovascular disease risk using clinical biomarkers.</strong>
</p>
## ✨ Features
<p align="center">
  <a href="#features"><img src="https://img.shields.io/badge/-Features-blueviolet?style=for-the-badge" alt="Features"/></a>
  <a href="#tech-stack"><img src="https://img.shields.io/badge/-Tech%20Stack-0d1117?style=for-the-badge" alt="Tech Stack"/></a>
  <a href="#getting-started"><img src="https://img.shields.io/badge/-Get%20Started-success?style=for-the-badge" alt="Get Started"/></a>
  <a href="#api-reference"><img src="https://img.shields.io/badge/-API%20Docs-blue?style=for-the-badge" alt="API Docs"/></a>
</p>
- **AI Prediction**: 90%+ ROC-AUC accuracy heart disease risk prediction
- **5 ML Models Compared**: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM
- **Premium Dark UI**: Glassmorphism, animations, ECG waveform, beating heart
- **Multi-Step Form**: Custom sliders, pill selectors, real-time validation
- **Risk Visualization**: Circular risk gauge, color-coded result cards
- **Health Tips**: Evidence-based cardiovascular health recommendations
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white" alt="Vite"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0-blue?logo=xgboost" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>
## 🚀 Quick Start
---
## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [ML Pipeline](#ml-pipeline)
- [API Reference](#api-reference)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)
---
## Overview
**HeartSense AI** is a full-stack cardiovascular risk assessment application. It combines a robust ML training pipeline (featuring automated model selection across 5 algorithms) with a sleek, dark-themed React dashboard that delivers instant, probability-based risk predictions from 11 clinical biomarkers.
Users enter clinical data — such as age, blood pressure, cholesterol, and ECG results — and the trained model returns a heart disease probability along with a confidence score and visual risk gauge.
---
## Features
|
 Category 
|
 Details 
|
|
---
|
---
|
|
 🤖 
**
Automated ML Pipeline
**
|
 Compares Logistic Regression, Random Forest, XGBoost, Gradient Boosting & SVM via GridSearchCV; auto-selects the best model 
|
|
 📊 
**
Advanced Preprocessing
**
|
 IQR-based outlier removal, SMOTE for class imbalance, feature engineering (age², HR/BP ratio, interaction terms) 
|
|
 ⚡ 
**
Real-time Predictions
**
|
 Flask REST API returns risk probability in milliseconds 
|
|
 🎨 
**
Premium Dark UI
**
|
 Glassmorphism cards, animated ECG line, heartbeat animations, risk gauge with SVG gradients 
|
|
 🔒 
**
Privacy-First
**
|
 No data is stored — all predictions happen in-memory on the server 
|
|
 ✅ 
**
Input Validation
**
|
 Server-side range checks on all 11 clinical fields with descriptive error messages 
|
|
 📱 
**
Fully Responsive
**
|
 Optimized for desktop, tablet, and mobile viewports 
|
---
## Architecture
```
┌──────────────────────┐         HTTP (JSON)        ┌──────────────────────────┐
│                      │  ──────────────────────►   │                          │
│   React + Vite       │   POST /predict            │   Flask REST API         │
│   (Frontend)         │   GET  /model-info         │   (Backend)              │
│                      │   GET  /health             │                          │
│   Port 5173          │  ◄──────────────────────   │   Port 5000              │
└──────────────────────┘                            └────────────┬─────────────┘
                                                                 │
                                                    ┌────────────▼─────────────┐
                                                    │   ML Model Artifacts     │
                                                    │   • trained_model.pkl    │
                                                    │   • scaler.pkl           │
                                                    │   • columns.pkl          │
                                                    │   • metadata.pkl         │
                                                    └──────────────────────────┘
```
---
## Tech Stack
### Backend
|
 Technology 
|
 Purpose 
|
|
---
|
---
|
|
**
Python 3.10+
**
|
 Core language 
|
|
**
Flask 3.0
**
|
 REST API framework 
|
|
**
Scikit-learn 1.3
**
|
 ML algorithms & evaluation 
|
|
**
XGBoost 2.0
**
|
 Gradient boosting classifier 
|
|
**
Imbalanced-learn
**
|
 SMOTE oversampling 
|
|
**
Pandas / NumPy
**
|
 Data manipulation 
|
|
**
Joblib
**
|
 Model serialization 
|
### Frontend
|
 Technology 
|
 Purpose 
|
|
---
|
---
|
|
**
React 19
**
|
 UI library 
|
|
**
Vite 8
**
|
 Build tool & dev server 
|
|
**
Axios
**
|
 HTTP client 
|
|
**
Tailwind CSS 4
**
|
 Utility-first styling 
|
|
**
Vanilla CSS
**
|
 Custom animations & glassmorphism 
|
---
## Project Structure
```
HeartMLProject/
├── backend/
│   ├── app.py                 # Flask API — prediction, model-info & health endpoints
│   ├── train_model.py         # Full ML pipeline — train, compare, select & save
│   ├── requirements.txt       # Python dependencies
│   ├── data/
│   │   └── heart.csv          # UCI Heart Disease dataset (918 samples)
│   ├── models/
│   │   ├── trained_model.pkl  # Best serialized model
│   │   ├── scaler.pkl         # StandardScaler fitted on training data
│   │   ├── columns.pkl        # Feature column order
│   │   └── metadata.pkl       # Model performance metadata
│   └── utils/
│       ├── __init__.py
│       └── preprocess.py      # Data cleaning, encoding & feature engineering
│
├── frontend/
│   ├── index.html             # Entry HTML
│   ├── package.json           # Node dependencies & scripts
│   ├── vite.config.js         # Vite configuration
│   ├── postcss.config.js      # PostCSS plugins
│   └── src/
│       ├── main.jsx           # React entry point
│       ├── App.jsx            # Main application component
│       ├── index.css           # Global styles, animations & design system
│       └── services/
│           └── api.js         # Axios API client
│
└── README.md
```
---
## Getting Started
### Prerequisites
- Python 3.8+
- Node.js 16+
- npm
### 1. Backend Setup
- **Python** 3.10 or higher — [Download](https://www.python.org/downloads/)
- **Node.js** 18+ & npm — [Download](https://nodejs.org/)
- **Git** — [Download](https://git-scm.com/)
### 1. Clone the Repository
```bash
git clone https://github.com/Bimleshyadav058/Heartdiseases.git
cd Heartdiseases
```
### 2. Backend Setup
```bash
# Navigate to backend
cd backend
# Create & activate virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
python train_model.py     # Train the ML model
python app.py             # Start Flask API on port 5000
```
### 2. Frontend Setup
#### Train the Model (first time only)
```bash
python train_model.py
```
This will:
- Load the UCI heart disease dataset
- Remove outliers using IQR method
- Engineer features (age², HR/BP ratio, interaction terms)
- Apply SMOTE for class balancing
- Train & tune 5 models via GridSearchCV
- Save the best model to `models/`
#### Start the API Server
```bash
python app.py
```
The API will be available at **`http://localhost:5000`**
### 3. Frontend Setup
```bash
# Open a new terminal and navigate to frontend
cd frontend
# Install dependencies
npm install
npm run dev               # Start Vite dev server on port 5173
# Start the dev server
npm run dev
```
### 3. Open the App
Navigate to `http://localhost:5173` in your browser.
The app will be available at **`http://localhost:5173`**
## 📁 Project Structure
---
## ML Pipeline
### Training Workflow
```
HeartMLProject/
├── backend/
│   ├── app.py                 # Flask REST API
│   ├── train_model.py         # ML training pipeline
│   ├── requirements.txt       # Python dependencies
│   ├── models/                # Saved model artifacts
│   │   ├── trained_model.pkl
│   │   ├── scaler.pkl
│   │   ├── columns.pkl
│   │   └── metadata.pkl
│   ├── data/
│   │   └── heart.csv          # Training dataset
│   └── utils/
│       └── preprocess.py      # Data preprocessing
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── animations/        # Framer Motion variants
│   │   ├── services/          # API service (Axios)
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   └── package.json
└── README.md
 CSV Data ──► Outlier Removal ──► Feature Engineering ──► Train/Test Split
                                         │
                                         ▼
                                  StandardScaler ──► SMOTE ──► GridSearchCV
                                                                    │
                                                                    ▼
                                                         ┌──────────────────┐
                                                         │  5 Models Tuned  │
                                                         │                  │
                                                         │  • Logistic Reg  │
                                                         │  • Random Forest │
                                                         │  • XGBoost      │
                                                         │  • Grad Boost   │
                                                         │  • SVM          │
                                                         └────────┬─────────┘
                                                                  │
                                                                  ▼
                                                         Best Model (by ROC-AUC)
                                                              saved to disk
```
## 🧠 ML Pipeline
### Feature Engineering
1. **Data**: UCI Heart Disease dataset (918 samples, 11 features)
2. **Preprocessing**: Missing value handling, outlier removal, feature engineering
3. **SMOTE**: Handles class imbalance
4. **GridSearchCV**: Hyperparameter tuning for each model
5. **Cross-Validation**: 5-fold stratified CV
6. **Best Model Selection**: Automatically selects by ROC-AUC score
|
 Engineered Feature 
|
 Formula 
|
 Rationale 
|
|
---
|
---
|
---
|
|
`Age_Squared`
|
 Age² 
|
 Captures non-linear age-risk relationship 
|
|
`HR_BP_Ratio`
|
 MaxHR / (RestingBP + 1) 
|
 Cardiac efficiency indicator 
|
|
`Age_HR`
|
 Age × MaxHR 
|
 Age-adjusted heart rate capacity 
|
|
`Oldpeak_Slope`
|
 Oldpeak × ST_Slope_Encoded 
|
 ST depression severity weighted by slope direction 
|
## 🔌 API Endpoints
### Evaluation Metrics
|
 Method 
|
 Endpoint 
|
 Description 
|
|
--------
|
----------
|
-------------
|
|
 POST 
|
`/predict`
|
 Predict heart disease risk 
|
|
 GET 
|
`/model-info`
|
 Get model metrics and info 
|
|
 GET 
|
`/health`
|
 Health check 
|
The pipeline evaluates each model on:
- **Accuracy** — Overall correct predictions
- **Precision** — Positive predictive value
- **Recall** — Sensitivity / true positive rate
- **F1 Score** — Harmonic mean of precision & recall
- **ROC-AUC** — Area under the receiver operating characteristic curve *(primary selection metric)*
- **5-Fold Stratified Cross-Validation**
## 🎨 Tech Stack
---
**Frontend**: React 18, Vite, Tailwind CSS, Framer Motion, Axios, Lucide Icons
**Backend**: Flask, Flask-CORS, Scikit-learn, XGBoost, Pandas, NumPy, Joblib
**ML**: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM
## API Reference
## ⚠️ Disclaimer
### `POST /predict`
This is an educational project and NOT a medical device. Always consult a qualified healthcare professional for medical advice.
# Heartdiseases
Predict heart disease risk from clinical biomarkers.
**Request Body:**
```json
{
  "age": 55,
  "sex": "M",
  "chestPainType": "ATA",
  "restingBP": 130,
  "cholesterol": 250,
  "fastingBS": 0,
  "restingECG": "Normal",
  "maxHR": 150,
  "exerciseAngina": "N",
  "oldpeak": 1.5,
  "stSlope": "Flat"
}
```
**Response (200):**
```json
{
  "prediction": "HIGH RISK",
  "probability": 73.2,
  "confidence": 73.2,
  "details": {
    "heart_disease_prob": 73.2,
    "healthy_prob": 26.8
  }
}
```
**Validation Constraints:**
|
 Field 
|
 Type 
|
 Valid Range / Values 
|
|
---
|
---
|
---
|
|
`age`
|
 integer 
|
 1 – 120 
|
|
`sex`
|
 string 
|
`"M"`
 or 
`"F"`
|
|
`chestPainType`
|
 string 
|
`"TA"`
, 
`"ATA"`
, 
`"NAP"`
, 
`"ASY"`
|
|
`restingBP`
|
 integer 
|
 50 – 300 (mmHg) 
|
|
`cholesterol`
|
 integer 
|
 50 – 700 (mg/dl) 
|
|
`fastingBS`
|
 integer 
|
`0`
 or 
`1`
|
|
`restingECG`
|
 string 
|
`"Normal"`
, 
`"ST"`
, 
`"LVH"`
|
|
`maxHR`
|
 integer 
|
 50 – 250 (bpm) 
|
|
`exerciseAngina`
|
 string 
|
`"Y"`
 or 
`"N"`
|
|
`oldpeak`
|
 float 
|
 -5.0 – 10.0 
|
|
`stSlope`
|
 string 
|
`"Up"`
, 
`"Flat"`
, 
`"Down"`
|
---
### `GET /model-info`
Returns performance metrics of the currently deployed model.
**Response (200):**
```json
{
  "model_name": "XGBoost",
  "accuracy": 90.5,
  "roc_auc": 95.2,
  "precision": 89.1,
  "recall": 93.4,
  "f1_score": 91.2,
  "features_count": 20,
  "training_samples": 1200
}
```
---
### `GET /health`
Health check endpoint.
```json
{
  "status": "healthy",
  "model_loaded": true
}
```
---
## Dataset
This project uses the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (combined from Cleveland, Hungarian, Switzerland & VA Long Beach), available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
- **Samples:** 918
- **Features:** 11 clinical attributes
- **Target:** Binary (0 = No Heart Disease, 1 = Heart Disease)
|
 Feature 
|
 Description 
|
|
---
|
---
|
|
 Age 
|
 Age in years 
|
|
 Sex 
|
 M = Male, F = Female 
|
|
 ChestPainType 
|
 TA, ATA, NAP, ASY 
|
|
 RestingBP 
|
 Resting blood pressure (mmHg) 
|
|
 Cholesterol 
|
 Serum cholesterol (mg/dl) 
|
|
 FastingBS 
|
 Fasting blood sugar > 120 mg/dl 
|
|
 RestingECG 
|
 Normal, ST, LVH 
|
|
 MaxHR 
|
 Maximum heart rate achieved 
|
|
 ExerciseAngina 
|
 Exercise-induced angina (Y/N) 
|
|
 Oldpeak 
|
 ST depression (numeric) 
|
|
 ST_Slope 
|
 Up, Flat, Down 
|
---
## Contributing
Contributions are welcome! Here's how to get started:
1. **Fork** the repository
2. **Create** a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit** your changes
   ```bash
   git commit -m "feat: add your feature description"
   ```
4. **Push** to your branch
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open** a Pull Request
### Ideas for Contribution
- 🧪 Add unit tests for the API and preprocessing pipeline
- 📈 Add training visualizations (confusion matrix heatmap, ROC curve plots)
- 🐳 Dockerize the application
- ☁️ Deploy to cloud (AWS / GCP / Heroku)
- 🌐 Add multi-language support to the frontend
- 📊 Add a model performance dashboard page
---
## License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
---
## Disclaimer
> ⚠️ **This application is for educational and research purposes only.** It is **not** a substitute for professional medical advice, diagnosis, or treatment. The predictions made by this model should not be used for clinical decision-making. Always consult a qualified healthcare provider for any medical concerns.
---
<p align="center">
  Made with ❤️ by <a href="https://github.com/Bimleshyadav058">Bimlesh Yadav</a>
</p>
#   H e a r t A I _ H e a r t _ D i s e a s e _ P r e  
 