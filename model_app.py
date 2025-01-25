from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (make sure to replace 'best_rf_model.joblib' with your model path)
model = joblib.load('house_price_predictor.joblib')

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request (expects a JSON object)
    data = request.get_json(force=True)
    
    # Convert input data to numpy array (model expects input as array)
    input_data = np.array(data['features']).reshape(1, -1)
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
