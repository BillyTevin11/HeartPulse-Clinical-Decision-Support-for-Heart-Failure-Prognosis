from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

# 1. LOAD ASSETS
# Ensure these filenames match your saved files from the training phase
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    logging.error("Model or Scaler files not found. Please ensure .pkl files are in the directory.")

# 2. DEFINE CLINICAL BASELINES
# Used for Local Explainability (comparing current patient to "typical" values)
# These represent the median values from the training dataset
BASELINES = {
    'age': 60, 'anaemia': 0, 'creatinine_phosphokinase': 250, 'diabetes': 0,
    'ejection_fraction': 38, 'high_blood_pressure': 0, 'platelets': 262000,
    'serum_creatinine': 1.1, 'serum_sodium': 137, 'sex': 1, 'smoking': 0, 'time': 115
}

# The exact order of features used during model training
FEATURE_ORDER = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
    'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 
    'smoking', 'time'
]

@app.route('/')
def home():
    """Renders the dashboard UI."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives patient data, processes features, calculates mortality risk, 
    and determines the top drivers for that specific patient.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided', 'status': 'failed'}), 400

        # Convert to DataFrame and enforce column order
        input_df = pd.DataFrame([data])
        input_df = input_df[FEATURE_ORDER] 
        
        # 1. PREPROCESS
        # Apply the same Log Transformation used during training
        skewed_cols = ['creatinine_phosphokinase', 'serum_creatinine']
        processed_df = input_df.copy()
        for col in skewed_cols:
            processed_df[col] = np.log1p(processed_df[col])
        
        # Scale features
        input_scaled = scaler.transform(processed_df)
        
        # 2. GET BASE PROBABILITY
        base_prob = model.predict_proba(input_scaled)[0][1]
        
        # 3. CALCULATE LOCAL EXPLAINABILITY (Impact Analysis)
        # We calculate the "Counterfactual Impact": how much the risk changes 
        # if a specific feature was changed to its baseline (neutral) value.
        contributions = []
        for col in FEATURE_ORDER:
            temp_df = processed_df.copy()
            
            # Reset the specific feature to baseline
            baseline_val = BASELINES[col]
            if col in skewed_cols:
                baseline_val = np.log1p(baseline_val)
            
            temp_df[col] = baseline_val
            temp_scaled = scaler.transform(temp_df)
            
            # Probability without this factor
            prob_without_feature = model.predict_proba(temp_scaled)[0][1]
            
            # Impact = current risk - risk without this factor
            impact = base_prob - prob_without_feature
            contributions.append({
                'feature': col.replace('_', ' ').title(),
                'impact': float(impact)
            })

        # Sort by absolute magnitude to identify top 5 risk/protective drivers
        contributions = sorted(contributions, key=lambda x: abs(x['impact']), reverse=True)

        return jsonify({
            'probability_score': round(float(base_prob), 4),
            'risk_level': "High Risk" if base_prob >= 0.35 else "Low Risk",
            'mortality_prediction': 1 if base_prob >= 0.35 else 0,
            'explanations': contributions[:5],
            'status': 'success'
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'message': str(e), 'status': 'failed'}), 500

if __name__ == '__main__':
    # Threaded=True allows handling multiple clinical requests simultaneously
    app.run(debug=False, port=5000, threaded=True)