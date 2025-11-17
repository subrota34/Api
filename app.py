# app.py - Fake News Detection API (Render.com + Flask 3.0 Ready)
# Works perfectly with your requirements.txt (Flask 3.0, scikit-learn 1.5.1, etc.)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

# ==================== Flask App Setup ====================
app = Flask(__name__)
CORS(app)  # Allow requests from Flutter (FlutLab)

# Paths for model and vectorizer (saved in Render's filesystem)
MODEL_PATH = '/tmp/fake_news_model.joblib'        # Render allows writing to /tmp
VECTORIZER_PATH = '/tmp/tfidf_vectorizer.joblib'
DATASET_PATH = 'news_articles.csv'                # Must be in same folder

# ==================== Model Training Function ====================
def train_and_save_model():
    print("Loading dataset and training model for the first time...")

    # Load your dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)

    # Clean data
    data['text'] = data['text'].fillna('')
    data = data.dropna(subset=['label'])

    X = data['text']
    y = data['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 2)  # Better detection with bigrams
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Passive Aggressive Classifier (best for fake news)
    model = PassiveAggressiveClassifier(max_iter=100, random_state=42, warm_start=True)
    model.fit(X_train_vec, y_train)

    # Test accuracy
    pred = model.predict(X_test_vec)
    accuracy = (pred == y_test).mean()
    print(f"Model trained successfully! Accuracy: {accuracy:.4f}")

    # Save model and vectorizer
    dump(model, MODEL_PATH)
    dump(vectorizer, VECTORIZER_PATH)
    print("Model and vectorizer saved to /tmp/")

# ==================== Load or Train Model ====================
print("Checking for saved model...")
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print("Loading pre-trained model...")
    model = load(MODEL_PATH)
    vectorizer = load(VECTORIZER_PATH)
    print("Model loaded successfully!")
else:
    print("No pre-trained model found. Training now...")
    train_and_save_model()
    model = load(MODEL_PATH)
    vectorizer = load(VECTORIZER_PATH)

# ==================== API Routes ====================

@app.route('/')
def home():
    return jsonify({
        "message": "Fake News Detection API is LIVE!",
        "status": "success",
        "model": "PassiveAggressiveClassifier + TF-IDF",
        "endpoint": "/predict",
        "method": "POST"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        json_data = request.get_json()
        if not json_data or 'text' not in json_data:
            return jsonify({"error": "No text provided in JSON body"}), 400

        text = json_data['text'].strip()
        if len(text) < 20:
            return jsonify({"error": "Text too short. Please enter a full news article."}), 400

        # Transform and predict
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        confidence = max(model.decision_function(text_vec)[0])  # Rough confidence

        return jsonify({
            "prediction": prediction,
            "confidence": round(float(confidence), 3),
            "text_length": len(text),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== Run App (Render + Gunicorn) ====================
if __name__ == '__main__':
    # For local testing only
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
else:
    # For Render + Gunicorn (required)
    pass