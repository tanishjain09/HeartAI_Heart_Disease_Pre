"""
Heart Disease Prediction - Flask API
=====================================
REST API for heart disease prediction using the trained ML model.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Load Model Artifacts ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

try:
    model = joblib.load(os.path.join(MODELS_DIR, 'trained_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    columns = joblib.load(os.path.join(MODELS_DIR, 'columns.pkl'))
    metadata = joblib.load(os.path.join(MODELS_DIR, 'metadata.pkl'))
    print("Model artifacts loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model artifacts: {e}")
    print("Please run train_model.py first.")
    model, scaler, columns, metadata = None, None, None, None


# Mapping for categorical features
CHEST_PAIN_MAP = {'TA': 'TA', 'ATA': 'ATA', 'NAP': 'NAP', 'ASY': 'ASY'}
RESTING_ECG_MAP = {'Normal': 'Normal', 'ST': 'ST', 'LVH': 'LVH'}
ST_SLOPE_MAP = {'Up': 'Up', 'Flat': 'Flat', 'Down': 'Down'}


def validate_input(data):
    """Validate incoming prediction request data."""
    required_fields = [
        'age', 'sex', 'chestPainType', 'restingBP', 'cholesterol',
        'fastingBS', 'restingECG', 'maxHR', 'exerciseAngina', 'oldpeak', 'stSlope'
    ]
    errors = []
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if not errors:
        try:
            age = int(data['age'])
            if not (1 <= age <= 120):
                errors.append("Age must be between 1 and 120")
        except (ValueError, TypeError):
            errors.append("Age must be a valid number")

        try:
            rbp = int(data['restingBP'])
            if not (50 <= rbp <= 300):
                errors.append("Resting BP must be between 50 and 300")
        except (ValueError, TypeError):
            errors.append("Resting BP must be a valid number")

        try:
            chol = int(data['cholesterol'])
            if not (50 <= chol <= 700):
                errors.append("Cholesterol must be between 50 and 700")
        except (ValueError, TypeError):
            errors.append("Cholesterol must be a valid number")

        try:
            mhr = int(data['maxHR'])
            if not (50 <= mhr <= 250):
                errors.append("Max HR must be between 50 and 250")
        except (ValueError, TypeError):
            errors.append("Max HR must be a valid number")

        try:
            op = float(data['oldpeak'])
            if not (-5 <= op <= 10):
                errors.append("Oldpeak must be between -5 and 10")
        except (ValueError, TypeError):
            errors.append("Oldpeak must be a valid number")

    return errors


def prepare_features(data):
    """Convert raw input data to model-ready feature vector."""
    age = int(data['age'])
    sex = 1 if data['sex'] == 'M' else 0
    resting_bp = int(data['restingBP'])
    cholesterol = int(data['cholesterol'])
    fasting_bs = int(data['fastingBS'])
    max_hr = int(data['maxHR'])
    exercise_angina = 1 if data['exerciseAngina'] == 'Y' else 0
    oldpeak = float(data['oldpeak'])

    # ST Slope encoding
    slope_map = {'Down': 0, 'Flat': 1, 'Up': 2}
    st_slope_encoded = slope_map.get(data['stSlope'], 1)

    # Feature engineering (must match training)
    age_squared = age ** 2
    hr_bp_ratio = max_hr / (resting_bp + 1)
    age_hr = age * max_hr
    oldpeak_slope = oldpeak * st_slope_encoded

    # One-hot encoding for ChestPainType
    cp = data['chestPainType']
    cp_ata = 1 if cp == 'ATA' else 0
    cp_asy = 1 if cp == 'ASY' else 0
    cp_nap = 1 if cp == 'NAP' else 0
    cp_ta = 1 if cp == 'TA' else 0

    # One-hot encoding for RestingECG
    ecg = data['restingECG']
    ecg_lvh = 1 if ecg == 'LVH' else 0
    ecg_normal = 1 if ecg == 'Normal' else 0
    ecg_st = 1 if ecg == 'ST' else 0

    # Build feature dict matching training columns
    feature_dict = {
        'Age': age,
        'Sex': sex,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'Age_Squared': age_squared,
        'HR_BP_Ratio': hr_bp_ratio,
        'Age_HR': age_hr,
        'ST_Slope_Encoded': st_slope_encoded,
        'Oldpeak_Slope': oldpeak_slope,
        'CP_ATA': cp_ata,
        'CP_ASY': cp_asy,
        'CP_NAP': cp_nap,
        'CP_TA': cp_ta,
        'ECG_LVH': ecg_lvh,
        'ECG_Normal': ecg_normal,
        'ECG_ST': ecg_st,
    }

    # Create DataFrame with correct column order
    df = pd.DataFrame([feature_dict])

    # Ensure all training columns are present
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]
    return df


@app.route('/predict', methods=['POST'])
def predict():
    """Predict heart disease risk from clinical biomarkers."""
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please run train_model.py first.'
        }), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Validate input
        errors = validate_input(data)
        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400

        # Prepare features
        features_df = prepare_features(data)
        features_scaled = scaler.transform(features_df)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"
        risk_probability = float(probability[1] * 100)

        return jsonify({
            'prediction': risk_label,
            'probability': round(risk_probability, 1),
            'confidence': round(max(probability) * 100, 1),
            'details': {
                'heart_disease_prob': round(float(probability[1] * 100), 1),
                'healthy_prob': round(float(probability[0] * 100), 1),
            }
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return information about the trained model."""
    if metadata is None:
        return jsonify({'error': 'Model metadata not available'}), 500

    return jsonify({
        'model_name': metadata.get('model_name', 'Unknown'),
        'accuracy': round(metadata.get('accuracy', 0) * 100, 1),
        'roc_auc': round(metadata.get('roc_auc', 0) * 100, 1),
        'precision': round(metadata.get('precision', 0) * 100, 1),
        'recall': round(metadata.get('recall', 0) * 100, 1),
        'f1_score': round(metadata.get('f1_score', 0) * 100, 1),
        'features_count': metadata.get('features_count', 0),
        'training_samples': metadata.get('training_samples', 0),
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
