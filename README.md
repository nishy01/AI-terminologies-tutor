📘 AI Terminology — Simple Definition Teller

A lightweight Streamlit web app that helps AI & Machine Learning students quickly understand technical terminology in plain English. Students can type any AI/ML term—even with spaces, abbreviations, or typos like back propagation, backprop, gradint descent, or conv net—to get an instant, simple definition.

✨ Features
Typo & Spacing Tolerant: Automatically matches noisy student inputs (back propagation → Backpropagation).
Real Datasets: Trained on 790 AI/ML terms & definitions from the Google Machine Learning Glossary cross-linked with 16,275 AI keyphrases from Hugging Face (J0nasW/AI-Dictionary).
Hybrid ML Model: Uses a Multi-Layer Perceptron (MLPClassifier) Neural Network + Character/Word N-Gram TF-IDF + Fuzzy Matching (97.28% training accuracy, 100% Top-1 typo benchmark accuracy).
📁 Files
app.py — Streamlit web application
train.py — Model training and prediction script
model.pkl — Pre-trained model artifact
ai_dictionary_dataset.csv — Combined AI/ML definitions dataset
AI_Dictionary_Colab.ipynb — Google Colab training & testing notebook
requirements.txt — Python dependencies
