import os
from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import string

# Initialize Flask app
app = Flask(__name__)

# Create required folders (one line each)
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Set upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your saved model and vectorizer
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Model and vectorizer loaded!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise e

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

# Prediction function
def predict_sentiment(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = max(proba)
    return pred, confidence

# Route: Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Predict on typed text
@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form.get('text', '').strip()
    if not user_text:  # ✅ Fixed: was broken
        return render_template('result.html', error="Please enter some text.")
    
    pred, conf = predict_sentiment(user_text)
    return render_template('result.html', text=user_text, sentiment=pred.upper(), confidence=f"{conf:.2f}")

# Route: Upload and analyze file
@app.route('/upload', methods=['POST'])
def upload_file():
    print("📤 Upload route triggered")
    if 'file' not in request.files:
        print("❌ No file in request")
        return render_template('index.html', error="No file part in request.")

    file = request.files['file']
    print(f"📄 File received: {file.filename}")

    if file.filename == '':
        print("❌ Empty filename")
        return render_template('index.html', error="No file selected.")

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    print(f"💾 Saving to: {filepath}")
    file.save(filepath)

    try:
        # Read file based on extension
        if file.filename.endswith('.csv'):
            # Import csv module for proper quoting constants
            import csv
            # Use pandas with proper quoting to handle commas in text fields
            df = pd.read_csv(filepath, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath, engine='openpyxl')
        else:
            return render_template('index.html', error="Unsupported file type. Use CSV or Excel (.csv, .xls, .xlsx)")

        # Validate content
        if df.empty:
            return render_template('index.html', error="Uploaded file is empty.")

        # Use first column as text
        text_column = df.columns[0]
        df[text_column] = df[text_column].astype(str)

        # Apply prediction
        results = []
        for idx, text in enumerate(df[text_column]):
            print(f"Processing row {idx+1}/{len(df)}")
            pred, conf = predict_sentiment(text)
            results.append({
                'original_text': text,
                'predicted_sentiment': pred,
                'confidence': conf
            })

        # Save result
        result_df = pd.DataFrame(results)
        output_path = 'results/output_with_sentiment.csv'  # Changed to CSV
        result_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)  # Use QUOTE_ALL to ensure all fields are quoted
        print(f"✅ Results saved to {output_path}")

        # Send file back
        return send_file(output_path, as_attachment=True)

    except Exception as e:
        print(f"❌ Error: {str(e)}")  # Show in terminal
        return render_template('index.html', error=f"Error processing file: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)