import unittest
import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, MODELS_DIR, prepare_features, validate_input

class TestHeartDiseaseModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load artifacts once for testing."""
        cls.model_path = os.path.join(MODELS_DIR, 'trained_model.pkl')
        cls.scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        cls.columns_path = os.path.join(MODELS_DIR, 'columns.pkl')
        cls.metadata_path = os.path.join(MODELS_DIR, 'metadata.pkl')
        
        cls.artifacts_exist = all(os.path.exists(p) for p in [
            cls.model_path, cls.scaler_path, cls.columns_path, cls.metadata_path
        ])
        
        if cls.artifacts_exist:
            cls.model = joblib.load(cls.model_path)
            cls.scaler = joblib.load(cls.scaler_path)
            cls.columns = joblib.load(cls.columns_path)
            cls.metadata = joblib.load(cls.metadata_path)

    def test_artifacts_exist(self):
        """Ensure all required model artifacts are generated and exist."""
        self.assertTrue(os.path.exists(self.model_path), "trained_model.pkl missing")
        self.assertTrue(os.path.exists(self.scaler_path), "scaler.pkl missing")
        self.assertTrue(os.path.exists(self.columns_path), "columns.pkl missing")
        self.assertTrue(os.path.exists(self.metadata_path), "metadata.pkl missing")

    def test_model_metadata(self):
        """Verify model metadata structure and performance targets."""
        if not self.artifacts_exist:
            self.skipTest("Model artifacts do not exist. Please train model first.")
            
        self.assertIn('model_name', self.metadata)
        self.assertIn('accuracy', self.metadata)
        self.assertIn('roc_auc', self.metadata)
        
        accuracy = self.metadata['accuracy']
        roc_auc = self.metadata['roc_auc']
        
        print(f"\n[Artifact Test] Model Name: {self.metadata['model_name']}")
        print(f"[Artifact Test] Model Accuracy: {accuracy * 100:.2f}%")
        print(f"[Artifact Test] Model ROC-AUC: {roc_auc:.4f}")
        
        # Check if accuracy meets baseline threshold
        self.assertGreaterEqual(accuracy, 0.75, "Model accuracy is below acceptable threshold (75%)")
        self.assertGreaterEqual(roc_auc, 0.80, "Model ROC-AUC score is below acceptable threshold (0.80)")

    def test_input_validation(self):
        """Test the input validation helper logic."""
        # Valid data
        valid_data = {
            'age': 55, 'sex': 'M', 'chestPainType': 'ASY', 'restingBP': 140,
            'cholesterol': 289, 'fastingBS': 0, 'restingECG': 'Normal',
            'maxHR': 150, 'exerciseAngina': 'N', 'oldpeak': 1.5, 'stSlope': 'Flat'
        }
        errors = validate_input(valid_data)
        self.assertEqual(len(errors), 0, f"Expected no validation errors, got: {errors}")
        
        # Missing fields
        invalid_data = {
            'age': 55, 'sex': 'M'
        }
        errors = validate_input(invalid_data)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Missing required field" in err for err in errors))
        
        # Out-of-bounds inputs
        out_of_bounds_data = valid_data.copy()
        out_of_bounds_data['age'] = 150  # Max is 120
        out_of_bounds_data['restingBP'] = 400  # Max is 300
        errors = validate_input(out_of_bounds_data)
        self.assertGreater(len(errors), 0)

    def test_feature_preparation(self):
        """Test preparation of features for model ingestion."""
        if not self.artifacts_exist:
            self.skipTest("Model artifacts do not exist.")
            
        test_input = {
            'age': 50, 'sex': 'F', 'chestPainType': 'ATA', 'restingBP': 120,
            'cholesterol': 210, 'fastingBS': 0, 'restingECG': 'LVH',
            'maxHR': 160, 'exerciseAngina': 'N', 'oldpeak': 0.0, 'stSlope': 'Up'
        }
        
        features_df = prepare_features(test_input)
        
        # Verify shape and column matches
        self.assertEqual(features_df.shape[0], 1, "Should return 1 row")
        self.assertEqual(list(features_df.columns), self.columns, "Columns structure should match trained features exactly")
        
        # Test feature engineering calculations
        self.assertEqual(features_df.loc[0, 'Age_Squared'], 2500)
        self.assertEqual(features_df.loc[0, 'Sex'], 0) # Female = 0
        self.assertEqual(features_df.loc[0, 'ExerciseAngina'], 0) # N = 0
        self.assertEqual(features_df.loc[0, 'ST_Slope_Encoded'], 2) # Up = 2

    def test_flask_endpoints(self):
        """Test public endpoints of Flask App via a test client."""
        client = app.test_client()
        
        # 1. Health check
        res_health = client.get('/health')
        self.assertEqual(res_health.status_code, 200)
        data_health = res_health.get_json()
        self.assertEqual(data_health['status'], 'healthy')
        
        # 2. Model Info
        res_info = client.get('/model-info')
        self.assertEqual(res_info.status_code, 200)
        data_info = res_info.get_json()
        self.assertIn('model_name', data_info)
        self.assertIn('accuracy', data_info)
        
        # 3. Predict Endpoint (Success case)
        test_payload = {
            'age': 58,
            'sex': 'M',
            'chestPainType': 'ASY',
            'restingBP': 136,
            'cholesterol': 248,
            'fastingBS': 0,
            'restingECG': 'LVH',
            'maxHR': 122,
            'exerciseAngina': 'Y',
            'oldpeak': 1.2,
            'stSlope': 'Flat'
        }
        res_pred = client.post('/predict', json=test_payload)
        self.assertEqual(res_pred.status_code, 200)
        data_pred = res_pred.get_json()
        self.assertIn('prediction', data_pred)
        self.assertIn('probability', data_pred)
        self.assertIn('confidence', data_pred)
        
        # 4. Predict Endpoint (Validation Failure case)
        bad_payload = test_payload.copy()
        bad_payload['age'] = -5  # Invalid age
        res_bad = client.post('/predict', json=bad_payload)
        self.assertEqual(res_bad.status_code, 400)
        data_bad = res_bad.get_json()
        self.assertIn('error', data_bad)
        self.assertIn('details', data_bad)

if __name__ == '__main__':
    unittest.main()
