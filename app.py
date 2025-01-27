from flask import Flask, request, jsonify
import joblib

# Load the model and vectorizer
model = joblib.load('newsgroups_text_classifier.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Initialize Flask app
app = Flask(__name__)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the input JSON data
    text = data['text']  # Extract the text from the input
    features = vectorizer.transform([text])  # Transform the text using the TF-IDF vectorizer
    prediction = model.predict(features)  # Make prediction
    category = newsgroups.target_names[prediction[0]]  # Map prediction to category name
    return jsonify({'category': category})  # Return the predicted category as JSON

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)