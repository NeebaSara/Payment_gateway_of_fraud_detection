from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the pre-trained models and encoder
with open('type_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('fraud_detection_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        transaction_type = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        
        # Encode the transaction type
        type_encoded = encoder.transform([transaction_type])[0]
        
        # Create feature array
        features = np.array([[type_encoded, amount, oldbalanceOrg, newbalanceOrig, 
                             oldbalanceDest, newbalanceDest]])
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        # Prepare response
        result = {
            'prediction': 'Fraudulent' if prediction[0] == 1 else 'Legitimate',
            'confidence': float(probability[0][prediction[0]] * 100),
            'fraud_probability': float(probability[0][1] * 100),
            'legitimate_probability': float(probability[0][0] * 100)
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)