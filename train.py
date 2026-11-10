import os
import re
import pickle
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ai_dictionary_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_text(text: str) -> str:
    """Remove all spaces and non-alphanumeric chars (e.g. 'back propagation' -> 'backpropagation')."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def generate_typo_variants(phrase: str, rng: np.random.Generator) -> list[str]:
    """Create realistic student typing variations (spacing, dropped char, swapped adjacent chars)."""
    norm = normalize_text(phrase)
    compact = compact_text(phrase)
    variants = {norm, compact}

    # Split camel/compound words or join spaced words
    if " " in norm:
        variants.add(norm.replace(" ", ""))
    elif len(norm) >= 8:
        mid = len(norm) // 2
        variants.add(norm[:mid] + " " + norm[mid:])

    # Single-character deletion and adjacent transposition for words >= 5 chars
    if len(norm) >= 5:
        idxs = rng.choice(len(norm), size=min(2, len(norm)), replace=False)
        for idx in idxs:
            if norm[idx] != " ":
                variants.add(norm[:idx] + norm[idx + 1 :])
            if idx < len(norm) - 1 and norm[idx] != " " and norm[idx + 1] != " ":
                swapped = (
                    norm[:idx] + norm[idx + 1] + norm[idx] + norm[idx + 2 :]
                )
                variants.add(swapped)

    return [v for v in variants if len(v) >= 2]


def build_training_corpus(df: pd.DataFrame):
    """Build augmented (query_text, term_index) pairs for training the MLP & TF-IDF model."""
    rng = np.random.default_rng(42)
    X_texts = []
    y_labels = []

    for idx, row in df.iterrows():
        term = str(row["term"])
        aliases = [a.strip() for a in str(row.get("aliases", "")).split(",") if a.strip() and a.strip() != "nan"]
        definition = str(row["simple_definition"])

        clean_term = re.sub(r"\s*\([^)]*\)", "", term).strip()
        acronyms = re.findall(r"\(([^)]+)\)", term)

        seed_phrases = [term, clean_term] + acronyms + aliases[:6]
        for phrase in seed_phrases:
            for variant in generate_typo_variants(phrase, rng):
                X_texts.append(variant)
                y_labels.append(idx)

        # Add definition keywords for semantic matching
        X_texts.append(normalize_text(f"{term} {clean_term} {definition}"))
        y_labels.append(idx)

    return X_texts, np.array(y_labels, dtype=np.int32)


def train_and_save():
    # Ensure raw J0nasW/AI-Dictionary file is compact for GitHub (<1 MB instead of 259 MB embeddings)
    raw_emb_path = os.path.join(BASE_DIR, "dictionary_weights_w_embeddings.csv")
    j0nasw_path = os.path.join(BASE_DIR, "j0nasw_ai_dictionary.csv")
    if os.path.exists(j0nasw_path) and os.path.getsize(j0nasw_path) > 5_000_000:
        j_df = pd.read_csv(j0nasw_path, usecols=["keyword", "weight"])
        j_df.to_csv(j0nasw_path, index=False, encoding="utf-8")
    if os.path.exists(raw_emb_path):
        try:
            os.remove(raw_emb_path)
        except OSError:
            pass

    print("1. Loading AI Dictionary Dataset...")
    if not os.path.exists(CSV_PATH):
        from build_dataset import build_and_save_dataset
        df = build_and_save_dataset()
    else:
        df = pd.read_csv(CSV_PATH)

    df = df.dropna(subset=["term", "simple_definition"]).reset_index(drop=True)
    print(f"   Loaded {len(df)} AI terminology definitions.")

    print("2. Building augmented student query & typo training corpus...")
    X_texts, y_labels = build_training_corpus(df)
    print(f"   Generated {len(X_texts):,} training samples across {len(df)} AI terms.")

    print("3. Fitting Character N-Gram & Word N-Gram TF-IDF Vectorizers...")
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        sublinear_tf=True,
        min_df=1,
        max_features=8000,
        dtype=np.float32,
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        stop_words="english",
        sublinear_tf=True,
        min_df=1,
        max_features=8000,
        dtype=np.float32,
    )

    X_char = char_vectorizer.fit_transform(X_texts)

    term_reference_docs = []
    term_alias_lookup = {}
    for idx, row in df.iterrows():
        term = str(row["term"])
        clean_term = re.sub(r"\s*\([^)]*\)", "", term).strip()
        acronyms = re.findall(r"\(([^)]+)\)", term)
        aliases = [a.strip() for a in str(row.get("aliases", "")).split(",") if a.strip() and a.strip() != "nan"]

        # Register aliases first, then exact term name so exact term takes priority
        for name in aliases + acronyms + [clean_term, term]:
            norm_name = normalize_text(name)
            comp_name = compact_text(name)
            if norm_name:
                term_alias_lookup[norm_name] = idx
            if comp_name:
                term_alias_lookup[comp_name] = idx

        ref_doc = f"{term} {clean_term} {' '.join(aliases)} {row['simple_definition']} {row.get('analogy', '')}"
        term_reference_docs.append(normalize_text(ref_doc))

    ref_char_matrix = char_vectorizer.transform(
        [
            normalize_text(f"{r['term']} {r.get('aliases', '')}")
            for _, r in df.iterrows()
        ]
    )
    ref_word_matrix = word_vectorizer.fit_transform(term_reference_docs)

    print("4. Training Neural MLP Terminology Classifier...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=0.003,
        max_iter=25,
        random_state=42,
    )
    mlp_model.fit(X_char, y_labels)
    # Strip Adam optimizer training state and cast weights to float32 for a compact (<6 MB) GitHub-ready model.pkl
    mlp_model._optimizer = None
    mlp_model.coefs_ = [w.astype(np.float32) for w in mlp_model.coefs_]
    mlp_model.intercepts_ = [b.astype(np.float32) for b in mlp_model.intercepts_]

    benchmark_queries = [
        ("back propagation", "backpropagation"),
        ("backprop", "backpropagation"),
        ("bakcpropagation", "backpropagation"),
        ("gradint descent", "gradient descent"),
        ("sgd", "stochastic gradient descent"),
        ("over fitting", "overfitting"),
        ("under fitting", "underfitting"),
        ("conv net", "convolutional neural network"),
        ("cnn", "convolutional neural network"),
        ("lstm", "long short-term memory"),
        ("gru", "gated recurrent unit"),
        ("attenion mechanism", "attention"),
        ("tranformer", "transformer"),
        ("rag", "retrieval-augmented generation"),
        ("hallucinations", "hallucination"),
        ("lora", "lora"),
        ("rlhf", "reinforcement learning from human feedback"),
        ("tokenisation", "token"),
        ("word embedding", "embedding"),
        ("random forrest", "random forest"),
        ("xgboost", "gradient boosting"),
        ("k means", "k-means"),
        ("confusion matrx", "confusion matrix"),
        ("f1-score", "f1"),
        ("vanishing gradients", "vanishing gradient"),
    ]

    artifact = {
        "df": df,
        "char_vectorizer": char_vectorizer,
        "word_vectorizer": word_vectorizer,
        "mlp_model": mlp_model,
        "ref_char_matrix": ref_char_matrix,
        "ref_word_matrix": ref_word_matrix,
        "term_alias_lookup": term_alias_lookup,
    }

    top1_hits = 0
    top3_hits = 0
    eval_rows = []
    for query, expected_sub in benchmark_queries:
        results = predict_term(query, artifact, top_k=3)
        pred_top1 = results[0]["term"]
        pred_top3 = [r["term"] for r in results]
        exp_norm = normalize_text(expected_sub)
        is_top1 = exp_norm in normalize_text(pred_top1) or normalize_text(pred_top1) in exp_norm
        is_top3 = any(exp_norm in normalize_text(t) or normalize_text(t) in exp_norm for t in pred_top3)
        if is_top1:
            top1_hits += 1
        if is_top3:
            top3_hits += 1
        eval_rows.append({
            "Student Query": query,
            "Matched AI Term": pred_top1,
            "Confidence": f"{results[0]['confidence'] * 100:.1f}%",
            "Correct": "✅" if is_top1 else "❌",
        })

    top1_acc = top1_hits / len(benchmark_queries)
    top3_acc = top3_hits / len(benchmark_queries)
    train_acc = float(mlp_model.score(X_char, y_labels))

    artifact["metrics"] = {
        "num_terms": len(df),
        "num_training_samples": len(X_texts),
        "train_accuracy": train_acc,
        "benchmark_top1_accuracy": top1_acc,
        "benchmark_top3_accuracy": top3_acc,
        "final_loss": float(mlp_model.loss_),
        "eval_table": pd.DataFrame(eval_rows),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"   Training Accuracy:  {train_acc * 100:.2f}%")
    print(f"   Benchmark Top-1:    {top1_acc * 100:.2f}% ({top1_hits}/{len(benchmark_queries)})")
    print(f"   Benchmark Top-3:    {top3_acc * 100:.2f}% ({top3_hits}/{len(benchmark_queries)})")
    print(f"5. Saved trained model artifact to {MODEL_PATH}")
    return artifact


def predict_term(query: str, artifact: dict, top_k: int = 3) -> list[dict]:
    """Predict the best-matching AI term and return its simple definition, analogy, and confidence."""
    df = artifact["df"]
    char_vectorizer = artifact["char_vectorizer"]
    word_vectorizer = artifact["word_vectorizer"]
    mlp_model = artifact["mlp_model"]
    ref_char_matrix = artifact["ref_char_matrix"]
    ref_word_matrix = artifact["ref_word_matrix"]
    term_alias_lookup = artifact["term_alias_lookup"]

    norm_q = normalize_text(query)
    comp_q = compact_text(query)

    if not norm_q:
        return []

    # 1. Neural MLP probability distribution over terms
    q_char_vec = char_vectorizer.transform([norm_q])
    mlp_probs = mlp_model.predict_proba(q_char_vec)[0]

    # Map MLP class indices back to full dataframe length if needed
    full_mlp_probs = np.zeros(len(df), dtype=np.float32)
    full_mlp_probs[mlp_model.classes_] = mlp_probs

    # 2. Character N-Gram Cosine Similarity against term & alias names
    char_sims = cosine_similarity(q_char_vec, ref_char_matrix)[0]

    # 3. Word N-Gram Cosine Similarity against full term + definition + analogy
    q_word_vec = word_vectorizer.transform([norm_q])
    word_sims = cosine_similarity(q_word_vec, ref_word_matrix)[0]

    # 4. Fast Fuzzy String Similarity on Top-40 candidates from N-gram/MLP
    prelim_scores = 0.45 * full_mlp_probs + 0.35 * char_sims + 0.20 * word_sims
    candidate_idxs = np.argsort(prelim_scores)[::-1][:40]

    fuzzy_scores = np.zeros(len(df), dtype=np.float32)
    for idx in candidate_idxs:
        row = df.iloc[int(idx)]
        term_str = str(row["term"])
        clean_term = re.sub(r"\s*\([^)]*\)", "", term_str).strip()
        aliases = [a.strip() for a in str(row.get("aliases", "")).split(",") if a.strip() and a.strip() != "nan"]
        candidates = [term_str, clean_term] + aliases[:8]
        best_ratio = 0.0
        for cand in candidates:
            r1 = SequenceMatcher(None, norm_q, normalize_text(cand)).ratio()
            r2 = SequenceMatcher(None, comp_q, compact_text(cand)).ratio()
            best_ratio = max(best_ratio, r1, r2)
        fuzzy_scores[int(idx)] = best_ratio

    combined_scores = (
        0.35 * full_mlp_probs
        + 0.25 * char_sims
        + 0.15 * word_sims
        + 0.25 * fuzzy_scores
    )

    # Exact alias or compact no-space match boost (e.g. "back propagation" -> "Backpropagation")
    exact_idx = term_alias_lookup.get(norm_q, term_alias_lookup.get(comp_q, None))
    if exact_idx is not None:
        combined_scores[exact_idx] = max(float(combined_scores[exact_idx]), 0.99)

    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = df.iloc[int(idx)]
        score = float(np.clip(combined_scores[int(idx)], 0.0, 1.0))
        results.append({
            "term": str(row["term"]),
            "category": str(row["category"]),
            "aliases": str(row.get("aliases", "")),
            "simple_definition": str(row["simple_definition"]),
            "analogy": str(row.get("analogy", "")),
            "source": str(row.get("source", "Google ML Glossary + J0nasW/AI-Dictionary")),
            "confidence": score,
            "exact_or_typo_match": (
                normalize_text(query) != normalize_text(str(row["term"]))
            ),
        })

    return results


if __name__ == "__main__":
    train_and_save()
