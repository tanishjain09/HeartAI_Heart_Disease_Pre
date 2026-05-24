"""
Heart Disease ML Training Pipeline
===================================
Trains and compares multiple ML models for heart disease prediction.
Uses GridSearchCV, SMOTE, cross-validation, and selects the best model.
Targets 90%+ accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils.preprocess import generate_heart_dataset, preprocess_data


def remove_outliers(df, columns, factor=1.5):
    """Remove outliers using the IQR method."""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean


def train_and_evaluate():
    """Main training pipeline."""
    print("=" * 60)
    print("  HEART DISEASE PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # --- Step 1: Load or Generate Dataset ---
    print("\n[1/7] Loading dataset...")
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'heart.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"  Loaded dataset from {data_path}: {df.shape}")
    else:
        print("  No CSV found. Generating synthetic heart disease dataset...")
        df = generate_heart_dataset(n_samples=918)
        os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"  Generated dataset saved to {data_path}: {df.shape}")

    print(f"  Features: {list(df.columns[:-1])}")
    print(f"  Target distribution:\n{df['HeartDisease'].value_counts().to_string()}")

    # --- Step 2: Remove Outliers ---
    print("\n[2/7] Removing outliers...")
    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    df_clean = remove_outliers(df, numeric_cols, factor=1.5)
    print(f"  Rows before: {len(df)}, after: {len(df_clean)}")

    # --- Step 3: Preprocess ---
    print("\n[3/7] Preprocessing data...")
    X, y = preprocess_data(df_clean)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Engineered features: {list(X.columns)}")

    # --- Step 4: Train/Test Split + SMOTE ---
    print("\n[4/7] Splitting data and applying SMOTE...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"  Train set after SMOTE: {X_train_resampled.shape}")
    print(f"  Test set: {X_test_scaled.shape}")

    # --- Step 5: Define Models with Hyperparameter Grids ---
    print("\n[5/7] Training models with GridSearchCV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=2000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(
                random_state=42, eval_metric='logloss', use_label_encoder=False
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    }

    results = {}
    best_score = 0
    best_model_name = None
    best_model = None

    for name, config in models.items():
        print(f"\n  Training {name}...")
        grid = GridSearchCV(
            config['model'],
            config['params'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train_resampled, y_train_resampled)

        # Evaluate on test set
        y_pred = grid.best_estimator_.predict(X_test_scaled)
        y_prob = grid.best_estimator_.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Cross-validation score
        cv_scores = cross_val_score(grid.best_estimator_, X_train_resampled,
                                     y_train_resampled, cv=cv, scoring='accuracy')

        results[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'roc_auc': auc, 'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(), 'best_params': grid.best_params_
        }

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {auc:.4f}")
        print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"    Best Params: {grid.best_params_}")

        if acc > best_score:
            best_score = acc
            best_model_name = name
            best_model = grid.best_estimator_

    # --- Step 6: Best Model Report ---
    print("\n" + "=" * 60)
    print(f"  BEST MODEL: {best_model_name}")
    print(f"  Accuracy: {best_score:.4f}")
    print("=" * 60)

    y_pred_best = best_model.predict(X_test_scaled)
    y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"    {cm}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['No Disease', 'Disease']))

    final_accuracy = accuracy_score(y_test, y_pred_best)
    print(f"\n  Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.1f}%)")

    # --- Step 7: Save Model Artifacts ---
    print("\n[7/7] Saving model artifacts...")
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(models_dir, 'trained_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(list(X.columns), os.path.join(models_dir, 'columns.pkl'))

    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'accuracy': float(final_accuracy),
        'roc_auc': float(results[best_model_name]['roc_auc']),
        'precision': float(results[best_model_name]['precision']),
        'recall': float(results[best_model_name]['recall']),
        'f1_score': float(results[best_model_name]['f1']),
        'features_count': len(X.columns),
        'training_samples': len(X_train_resampled),
        'test_samples': len(X_test),
        'best_params': results[best_model_name]['best_params'],
        'all_results': {k: {mk: float(mv) if isinstance(mv, (np.floating, float)) else mv
                            for mk, mv in v.items()}
                        for k, v in results.items()}
    }
    joblib.dump(metadata, os.path.join(models_dir, 'metadata.pkl'))

    print(f"  Saved: trained_model.pkl, scaler.pkl, columns.pkl, metadata.pkl")
    print(f"  Location: {models_dir}")
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)

    return best_model, scaler, list(X.columns), metadata


if __name__ == '__main__':
    train_and_evaluate()
