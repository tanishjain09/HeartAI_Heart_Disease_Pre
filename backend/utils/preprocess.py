"""
Heart Disease Data Preprocessing Utilities
Handles data generation, cleaning, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np


def generate_heart_dataset(n_samples=918, random_state=42):
    """
    Generate a realistic synthetic heart disease dataset based on
    known medical distributions from the UCI/Kaggle Heart Disease dataset.
    Features are correlated with the target following established cardiology research.
    """
    np.random.seed(random_state)
    n = n_samples

    # Create latent risk score
    risk = np.random.randn(n)

    # Generate target based on risk score
    prob = 1 / (1 + np.exp(-1.2 * risk))
    target = (np.random.random(n) < prob).astype(int)

    # --- Generate features correlated with risk ---

    # Age: older patients have higher risk
    age = (54 + 8 * risk + np.random.randn(n) * 5).clip(28, 77).astype(int)

    # Sex: males have slightly higher risk
    sex_prob = np.clip(0.55 + 0.15 * risk, 0.2, 0.9)
    sex = np.where(np.random.random(n) < sex_prob, 'M', 'F')

    # Chest Pain Type: ASY most common for positive cases
    cp_choices = ['TA', 'ATA', 'NAP', 'ASY']
    cp = []
    for i in range(n):
        if target[i] == 1:
            cp.append(np.random.choice(cp_choices, p=[0.05, 0.10, 0.15, 0.70]))
        else:
            cp.append(np.random.choice(cp_choices, p=[0.15, 0.30, 0.35, 0.20]))

    # Resting Blood Pressure
    resting_bp = (130 + 7 * risk + np.random.randn(n) * 15).clip(90, 200).astype(int)

    # Cholesterol
    chol = (240 + 12 * risk + np.random.randn(n) * 45).clip(100, 600).astype(int)

    # Fasting Blood Sugar > 120 mg/dl
    fbs_prob = np.clip(0.15 + 0.1 * risk, 0.05, 0.5)
    fbs = (np.random.random(n) < fbs_prob).astype(int)

    # Resting ECG
    restecg_choices = ['Normal', 'ST', 'LVH']
    restecg = []
    for i in range(n):
        if target[i] == 1:
            restecg.append(np.random.choice(restecg_choices, p=[0.40, 0.40, 0.20]))
        else:
            restecg.append(np.random.choice(restecg_choices, p=[0.65, 0.20, 0.15]))

    # Max Heart Rate: lower for positive cases
    max_hr = (150 - 12 * risk + np.random.randn(n) * 18).clip(60, 202).astype(int)

    # Exercise Angina
    exang_prob = np.clip(0.3 + 0.25 * risk, 0.05, 0.85)
    exang = np.where(np.random.random(n) < exang_prob, 'Y', 'N')

    # Oldpeak (ST depression)
    oldpeak = np.round((0.5 + 0.8 * risk + np.random.randn(n) * 0.5).clip(-2.6, 6.2), 1)

    # ST Slope
    slope_choices = ['Up', 'Flat', 'Down']
    slope = []
    for i in range(n):
        if target[i] == 1:
            slope.append(np.random.choice(slope_choices, p=[0.15, 0.70, 0.15]))
        else:
            slope.append(np.random.choice(slope_choices, p=[0.60, 0.30, 0.10]))

    df = pd.DataFrame({
        'Age': age,
        'Sex': sex,
        'ChestPainType': cp,
        'RestingBP': resting_bp,
        'Cholesterol': chol,
        'FastingBS': fbs,
        'RestingECG': restecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exang,
        'Oldpeak': oldpeak,
        'ST_Slope': slope,
        'HeartDisease': target
    })

    return df


def preprocess_data(df):
    """
    Preprocess the heart disease dataset:
    - Handle missing values
    - Handle zero cholesterol
    - Feature engineering
    - Categorical encoding (one-hot)
    Returns: (X dataframe, y series)
    """
    df = df.copy()

    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'HeartDisease':
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Handle zero cholesterol (likely missing)
    if 'Cholesterol' in df.columns:
        median_chol = df[df['Cholesterol'] > 0]['Cholesterol'].median()
        df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = median_chol

    # Feature engineering
    df['Age_Squared'] = df['Age'] ** 2
    df['HR_BP_Ratio'] = df['MaxHR'] / (df['RestingBP'] + 1)
    df['Age_HR'] = df['Age'] * df['MaxHR']

    # Encode Sex
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})

    # Encode ExerciseAngina
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})

    # Encode ST_Slope (ordinal)
    slope_map = {'Down': 0, 'Flat': 1, 'Up': 2}
    df['ST_Slope_Encoded'] = df['ST_Slope'].map(slope_map)

    # Create interaction feature before dropping ST_Slope
    df['Oldpeak_Slope'] = df['Oldpeak'] * df['ST_Slope_Encoded']

    # One-hot encode ChestPainType
    cp_dummies = pd.get_dummies(df['ChestPainType'], prefix='CP', dtype=int)
    df = pd.concat([df, cp_dummies], axis=1)

    # One-hot encode RestingECG
    ecg_dummies = pd.get_dummies(df['RestingECG'], prefix='ECG', dtype=int)
    df = pd.concat([df, ecg_dummies], axis=1)

    # Drop original categorical columns
    df.drop(['ChestPainType', 'RestingECG', 'ST_Slope'], axis=1, inplace=True)

    # Separate features and target
    y = df['HeartDisease']
    X = df.drop('HeartDisease', axis=1)

    return X, y
