# 📘 AI Terminology — Simple Definition Teller (Streamlit Web App)

A fast, student-friendly **AI & Machine Learning Definition Teller** web application built with **Python, Scikit-Learn (Neural MLP + Character/Word N-Gram TF-IDF), and Streamlit**.

Instead of writing long prompts in a general chatbot, AI students can simply type a technical term—even with spaces, abbreviations, or typos like `back propagation`, `backprop`, `gradint descent`, or `conv net`—and immediately get a **clear, plain-English definition** and **additional intuition/context**.

---

## 📊 Real External Datasets Used

1. **Official Google Machine Learning Glossary (`developers.google.com/machine-learning/glossary`):**
   - **733+ official AI, Machine Learning, Deep Learning, NLP, Computer Vision, and Data Science terms & definitions** combined into **790 total indexed AI terms** (`ai_dictionary_dataset.csv`).
2. **Hugging Face `J0nasW/AI-Dictionary` (`huggingface.co/datasets/J0nasW/AI-Dictionary`):**
   - **16,275 real AI keyphrases** (`j0nasw_ai_dictionary.csv`) cross-referenced with our glossary terms to capture real-world noisy spellings and spacing variations (such as `graph neuralnetworks`, `back propagation`, etc.).

---

## 🧠 Model Architecture & Evaluation Metrics

- **Architecture:** Hybrid **Multi-Layer Perceptron (`MLPClassifier`, 128 → 64 ReLU layers)** + **Character N-Gram TF-IDF (`2–5 char_wb`)** + **Word N-Gram TF-IDF (`1–3 words`)** + **Fuzzy Sequence Alignment (`SequenceMatcher`)**.
- **Training Corpus:** **15,277 augmented student queries** (including spacing variants, abbreviations, and synthetic character typos across all 790 AI terms).
- **Training Accuracy:** **97.16%**
- **Student Typo & Spacing Benchmark (Top-1 Accuracy):** **100.00% (25/25)**
- **Student Typo & Spacing Benchmark (Top-3 Accuracy):** **100.00% (25/25)**

---

## 📁 Project Files

| File | Description |
| :--- | :--- |
| `app.py` | Main Streamlit web application |
| `train.py` | Trains the Hybrid Neural MLP + N-Gram TF-IDF model and saves `model.pkl` |
| `fetch_and_train.py` | Downloads the Official Google ML Glossary, links it with `J0nasW/AI-Dictionary`, builds `ai_dictionary_dataset.csv`, and trains the model |
| `download_hf_dataset.py` | Downloads `J0nasW/AI-Dictionary` (`16,275` rows) from Hugging Face Hub |
| `ai_dictionary_dataset.csv` | Combined dataset of **790 AI/ML terms, categories, aliases, and simple definitions** |
| `j0nasw_ai_dictionary.csv` | **16,275 AI keyphrases and weights** from Hugging Face `J0nasW/AI-Dictionary` |
| `model.pkl` | Pre-trained compact model artifact ready for GitHub & Streamlit Cloud |
| `requirements.txt` | Python dependencies for local run & Streamlit Cloud deployment |

---

## 🚀 How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit app
streamlit run app.py
```

---

## 🌐 Deploying to GitHub & Streamlit Community Cloud

1. Create a new repository on **GitHub** (e.g., `ai-dictionary-app`).
2. Upload the files from `C:\Users\nirja\.gemini\antigravity\scratch\ai-dictionary-app` (`app.py`, `train.py`, `ai_dictionary_dataset.csv`, `j0nasw_ai_dictionary.csv`, `model.pkl`, `requirements.txt`, `README.md`).
3. Go to [share.streamlit.io](https://share.streamlit.io/), click **New app**, select your GitHub repository, set **Main file path** to `app.py`, and click **Deploy!**
