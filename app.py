"""
Stroke Risk Prediction - Flask Web Application
Provides a REST API endpoint for stroke risk prediction
"""

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any

app = Flask(__name__)

# Load model and preprocessing objects
def load_model(filepath: str = 'stroke_model.pkl') -> Dict[str, Any]:
    """Load the trained model and preprocessing components"""
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        print("✅ Model loaded successfully!")
        return model_data
    except FileNotFoundError:
        print("❌ Model file not found. Please run train_model.py first.")
        raise

# Initialize model data
model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
features = model_data['feature_columns']
label_encoders = model_data.get('label_encoders', {})

# Mapping dictionaries for form inputs
WORK_TYPE_MAP = {
    'private': 'Private',
    'self-employed': 'Self-employed',
    'govt_job': 'Govt_job',
    'children': 'children',
    'never_worked': 'Never_worked'
}


def prepare_input_data(form_data: Dict) -> pd.DataFrame:
    """Convert form data to model input format"""
    # Use the loaded label encoders (this is CRITICAL!)
    gender_encoded = label_encoders['gender'].transform([form_data.get('gender', 'Female')])[0]
    work_type_str = WORK_TYPE_MAP.get(form_data.get('work_type', 'private'), 'Private')
    work_encoded = label_encoders['work_type'].transform([work_type_str])[0]
    married_encoded = label_encoders['ever_married'].transform([form_data.get('ever_married', 'No')])[0]
    residence_encoded = label_encoders['Residence_type'].transform([form_data.get('Residence_type', 'Rural')])[0]
    smoking_encoded = label_encoders['smoking_status'].transform([form_data.get('smoking_status', 'never smoked')])[0]
    
    input_dict = {
        'gender': gender_encoded,
        'age': float(form_data.get('age', 0)),
        'hypertension': int(form_data.get('hypertension', 0)),
        'heart_disease': int(form_data.get('heart_disease', 0)),
        'ever_married': married_encoded,
        'work_type': work_encoded,
        'Residence_type': residence_encoded,
        'avg_glucose_level': float(form_data.get('avg_glucose_level', 0)),
        'bmi': float(form_data.get('bmi', 0)),
        'smoking_status': smoking_encoded
    }
    
    return pd.DataFrame([[input_dict[f] for f in features]], columns=features)

def generate_recommendation(prediction: int, probability: float) -> str:
    """Generate health recommendation based on prediction"""
    if prediction == 1:
        if probability > 0.7:
            return "⚠️ High risk detected. Please consult a healthcare provider immediately for a comprehensive evaluation."
        else:
            return "⚠️ Moderate risk detected. Consider scheduling a checkup and discussing preventive measures with your doctor."
    else:
        if probability > 0.3:
            return "✅ Low risk detected. Continue healthy habits and maintain regular medical checkups."
        else:
            return "✅ Very low risk detected. Keep up with your healthy lifestyle!"

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict stroke risk based on patient data
    Expects form data with patient health metrics
    """
    try:
        # Get form data
        form_data = request.form
        
        # Validate required fields
        required_fields = ['age', 'avg_glucose_level', 'bmi']
        for field in required_fields:
            if not form_data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare input data
        input_df = prepare_input_data(form_data)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        probability = float(model.predict_proba(input_scaled)[0][1])
        prediction = int(probability > 0.5)
        
        # Generate response
        response = {
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'risk_level': 'High' if prediction == 1 else 'Low',
            'message': generate_recommendation(prediction, probability)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during prediction'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🧠 STROKE RISK PREDICTION SYSTEM")
    print("=" * 50)
    print("🌐 Starting Flask server...")
    print("📍 Visit: http://localhost:5000")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)