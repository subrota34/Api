# app.py → Works with Start Command: python app.py   (Render.com approved Nov 2025)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

app = Flask(__name__)
CORS(app)

# Paths (Render allows writing here)
MODEL_PATH = '/tmp/fake_news_model.joblib'
VECTORIZER_PATH = '/tmp/tfidf_vectorizer.joblib'
DATASET_PATH = 'news_articles.csv'

def train_and_save_model():
    print("First time launch → Training model...")
    data = pd.read_csv(DATASET_PATH)
    data['text'] = data['text'].fillna('')
    data = data.dropna(subset=['label'])

    X = data['text']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words='english', max_df=0.7, min_df=2, ngram_range=(1,2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    model = PassiveAggressiveClassifier(max_iter=100, random_state=42)
    model.fit(X_train_vec, y_train)

    # Save
    dump(model, MODEL_PATH)
    dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved! Accuracy on test: {(model.predict(vectorizer.transform(X_test)) == y_test).mean():.4f}")

# Load or train
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = load(MODEL_PATH)
    vectorizer = load(VECTORIZER_PATH)
    print("Pre-trained model loaded!")
else:
    train_and_save_model()
    model = load(MODEL_PATH)
    vectorizer = load(VECTORIZER_PATH)

@app.route('/')
def home():
    return jsonify({"message": "Fake News Detector API is LIVE (BD 2025)", "status": "OK"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json(force=True)
        text = json_data.get('text', '').strip()
        
        if not text or len(text) < 15:
            return jsonify({"error": "Please enter proper news text"}), 400

        vec = vectorizer.transform([text])
        result = model.predict(vec)[0]

        return jsonify({
            "prediction": result,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# THIS IS THE KEY PART FOR RENDER
if __name__ == '__main__':
    # Render gives PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    # These two lines make python app.py work perfectly on Render
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
