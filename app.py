from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re
import logging


loaded_model = joblib.load(r'C:\Users\user\Desktop\yolov5\spam message\weight\logistic_model.pkl')
vectorizer_path = joblib.load(r'C:\Users\user\Desktop\yolov5\spam message\weight\vectorizer_model.pkl')


logging.basicConfig(level=logging.INFO)


def extract_additional_features(text):
    urgent_keywords = ['urgent', 'call now', 'claim', 'secure', 'limited offer', 'compromised', 'verify', 'immediately']
    financial_keywords = ['free', 'lottery', 'prize', 'money', 'credit', 'loan', 'win', 'cash', 'offer']
    
    urgent_feature = any(keyword in text.lower() for keyword in urgent_keywords)
    financial_feature = any(keyword in text.lower() for keyword in financial_keywords)
    phone_number_feature = bool(re.search(r'\+?\d[\d -]{3,15}\d|\b\d{3,4}[- ]?\d{3,4}\b', text))
    
    return [int(urgent_feature), int(financial_feature), int(phone_number_feature)]

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')  


@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.form['message']
        logging.info(f"Input message: {user_input}")

        # Vectorize input
        user_input_vec = vectorizer_path.transform([user_input])
        logging.info(f"Vectorized input shape: {user_input_vec.shape}")
        logging.info(f"Vectorized input sample: {user_input_vec}")

        # Extract additional features
        user_input_additional = np.array([extract_additional_features(user_input)])
        logging.info(f"Additional features: {user_input_additional}")

        # Combine features
        user_input_combined = hstack([user_input_vec, user_input_additional])
        logging.info(f"Combined input shape: {user_input_combined.shape}")

        # Predict probabilities
        probabilities = loaded_model.predict_proba(user_input_combined)[:, 1]
        logging.info(f"Predicted probabilities: {probabilities}")

        # Apply threshold
        threshold = 0.1  # Try adjusting this
        prediction = (probabilities > threshold).astype(int)
        logging.info(f"Prediction: {prediction}")

        # Return result
        result = "spam" if prediction[0] == 1 else "ham"
        logging.info(f"Final result: {result}")
        return jsonify({'prediction': result})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/')
def about():
    return render_template('about.html')  

@app.route('/')
def contact():
    return render_template('contact.html')  



if __name__ == '__main__':
    app.run(debug=True)