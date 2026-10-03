import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the K-Means model
model = joblib.load('kmeans_model.joblib')
model = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return 'K-Means Model API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Assuming input data is in the format {'annual_income': X, 'spending_score': Y}
    annual_income = data['annual_income']
    spending_score = data['spending_score']
    
    # Make prediction
    prediction = model.predict(np.array([[annual_income, spending_score]]))
    
    return jsonify({'cluster': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
