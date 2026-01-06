# 📊 Sentiment Analysis Tool

A web-based AI tool that analyzes the sentiment of text — whether it's positive, negative, or neutral. Built with **Python, Flask, and Machine Learning**.

## 🔍 Overview

This app allows users to:
- ✅ Type a sentence and get instant sentiment prediction
- 📁 Upload a CSV/Excel file to analyze hundreds of reviews at once
- 💾 Download results with sentiment and confidence scores

Perfect for analyzing customer feedback, social media posts, product reviews, and more!

Built as an end-to-end machine learning pipeline: from data preprocessing to model deployment.

---

## 🚀 Features

| Feature | Description |
|-------|-------------|
| ✏️ Single Text Input | Analyze any sentence in real time |
| 📂 Batch File Upload | Supports `.csv`, `.xls`, `.xlsx` files |
| 📥 Auto-Download Results | Get a new Excel file with predictions |
| 🧠 ML-Powered | Trained on real Amazon reviews using Logistic Regression |
| 🌐 Responsive UI | Beautiful interface using Tailwind CSS & Feather Icons |

---

## 🛠️ Tech Stack

- **Backend**: Python + Flask
- **Machine Learning**: Scikit-learn (Logistic Regression + TF-IDF)
- **Frontend**: HTML, Tailwind CSS
- **Data Processing**: Pandas, Joblib
- **Deployment Ready**: Easy to deploy on Render, Railway, or Vercel

---

## ▶️ How to Run Locally

### Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-app.git
cd sentiment-analysis-app
