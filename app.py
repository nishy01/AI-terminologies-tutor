import os
import pickle
import streamlit as st
import pandas as pd
from train import predict_term, train_and_save, normalize_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

st.set_page_config(
    page_title="AI Terminology Simple Definition Teller",
    page_icon="📘",
    layout="centered",
)

# Custom styling for a clean, student-friendly dictionary card
st.markdown(
    """
    <style>
    .def-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 6px solid #38bdf8;
        border-radius: 12px;
        padding: 24px;
        margin-top: 12px;
        margin-bottom: 18px;
        color: #f8fafc;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .term-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .category-badge {
        display: inline-block;
        background-color: #334155;
        color: #e2e8f0;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .def-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 4px;
    }
    .def-text {
        font-size: 1.18rem;
        line-height: 1.6;
        color: #f1f5f9;
        margin-bottom: 14px;
    }
    .analogy-box {
        background-color: rgba(56, 189, 248, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 1.0rem;
        color: #bae6fd;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return train_and_save()
    try:
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        if "mlp_model" not in artifact or "df" not in artifact:
            return train_and_save()
        return artifact
    except Exception:
        return train_and_save()


artifact = load_trained_model()
df_terms = artifact["df"]
metrics = artifact.get("metrics", {})

# Header
st.title("📘 AI Terminology — Simple Definition Teller")
st.write(
    "Type any **AI, Machine Learning, or Deep Learning term** below (even with spaces or typos like "
    "`back propagation`, `backprop`, or `gradint descent`) to get an instant, plain-English definition built for students."
)
st.caption(
    f"📊 **Trained on {len(df_terms):,} Official AI/ML Terms** from the **Google Machine Learning Glossary** "
    "cross-linked with **16,275 keyphrases from Hugging Face `J0nasW/AI-Dictionary`**."
)

# Quick example buttons
st.caption("💡 **Quick try (click any term):**")
quick_terms = [
    "back propagation",
    "gradient descent",
    "overfitting",
    "transformer",
    "conv net",
    "RAG",
    "hallucination",
    "epoch",
]

if "student_query" not in st.session_state:
    st.session_state.student_query = "back propagation"

cols = st.columns(4)
for i, sample in enumerate(quick_terms):
    if cols[i % 4].button(sample, key=f"btn_{sample}", use_container_width=True):
        st.session_state.student_query = sample

# Search Input Box
query = st.text_input(
    "🔍 Enter an AI / ML Term:",
    value=st.session_state.student_query,
    placeholder="e.g., back propagation, overfitting, attention mechanism, LSTM, F1 score...",
)

if query.strip():
    predictions = predict_term(query, artifact, top_k=3)
    if predictions:
        top = predictions[0]
        confidence_pct = top["confidence"] * 100

        # Show typo / alias auto-match notification if query differs from canonical term
        if normalize_text(query) != normalize_text(top["term"]):
            st.info(
                f"✨ **Fuzzy / Typo Match:** Matched your input **`\"{query.strip()}\"`** "
                f"→ **`{top['term']}`** ({confidence_pct:.0f}% match score)"
            )

        analogy_html = ""
        if top.get("analogy") and str(top["analogy"]).strip() and str(top["analogy"]) != "nan":
            analogy_html = f"""
                <div class="def-label">💡 Intuition / Additional Context</div>
                <div class="analogy-box">{top['analogy']}</div>
            """

        # Render the Simple Definition Card
        st.markdown(
            f"""
            <div class="def-card">
                <div class="term-title">{top['term']}</div>
                <span class="category-badge">🏷️ {top['category']} &nbsp;|&nbsp; 🎯 Confidence: {confidence_pct:.0f}%</span>
                <div class="def-label">📖 Simple Student Definition</div>
                <div class="def-text">{top['simple_definition']}</div>
                {analogy_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"**Aliases / J0nasW Keyphrases:** `{top['aliases']}`  \n"
            f"**Dataset Source:** `{top.get('source', 'Google ML Glossary + J0nasW/AI-Dictionary')}`"
        )

        # Show other close matches in case the student was looking for a related term
        if len(predictions) > 1:
            with st.expander("🔎 See Top 3 Closest Matching Terms from the Model"):
                for rank, item in enumerate(predictions, start=1):
                    st.markdown(
                        f"**{rank}. {item['term']}** *({item['category']} — {item['confidence']*100:.0f}% score)*  \n"
                        f"{item['simple_definition']}"
                    )
else:
    st.info("👆 Type an AI term above (like `back propagation`) to see its simple definition.")

st.divider()

# Bottom tabs for Model Details & Full AI Dictionary Explorer
tab1, tab2 = st.tabs(["📚 Browse Full AI Glossary Dataset", "🧠 Model Architecture & Accuracy"])

with tab1:
    st.subheader(f"All {len(df_terms):,} AI/ML Terms in the Dataset")
    categories = ["All Categories"] + sorted(df_terms["category"].dropna().unique().tolist())
    selected_cat = st.selectbox("Filter by AI Domain:", categories)

    if selected_cat == "All Categories":
        filtered_df = df_terms
    else:
        filtered_df = df_terms[df_terms["category"] == selected_cat]

    display_cols = [c for c in ["term", "category", "simple_definition", "aliases", "source"] if c in filtered_df.columns]
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("How the AI Definition Teller Model Works")
    st.markdown(
        """
        1. **Real External Datasets Combined:**
           - **Official Google Machine Learning Glossary (`developers.google.com/machine-learning/glossary`):** Provides 450+ authoritative, beginner-friendly definitions of AI, Machine Learning, Deep Learning, Generative AI, and Data Science terms.
           - **Hugging Face `J0nasW/AI-Dictionary` (`16,275` AI keyphrases):** Cross-linked to provide real-world spelling/spacing variants (such as `graph neuralnetworks`, `back propagation`, etc.).
        2. **Character N-Gram TF-IDF Vectorizer (`2–5 char_wb`):** Breaks input text into overlapping character chunks so spacing differences (`back propagation` vs `backpropagation`) and spelling typos (`bakcpropagation`, `gradint descent`) still produce nearly identical feature vectors.
        3. **Multi-Layer Perceptron (`MLPClassifier` Neural Network):** A 2-hidden-layer `(128, 64)` ReLU neural network trained on **augmented student queries, aliases, abbreviations, and synthetic typos**.
        4. **Word N-Gram + Fuzzy Sequence Alignment (`SequenceMatcher`):** Combines neural probabilities with character/word cosine similarity and edit-distance ratio for high precision.
        """
    )

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AI Terms Indexed", f"{metrics.get('num_terms', len(df_terms)):,}")
        c2.metric("Augmented Training Queries", f"{metrics.get('num_training_samples', 0):,}")
        c3.metric("Typo Benchmark (Top-1)", f"{metrics.get('benchmark_top1_accuracy', 1.0)*100:.1f}%")
        c4.metric("Typo Benchmark (Top-3)", f"{metrics.get('benchmark_top3_accuracy', 1.0)*100:.1f}%")

        if "eval_table" in metrics:
            st.write("**Sample Evaluation on Student Typos, Spacing & Abbreviations:**")
            st.dataframe(metrics["eval_table"], use_container_width=True, hide_index=True)
