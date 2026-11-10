import os
import re
import json
import html
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
J0NASW_CSV_PATH = os.path.join(BASE_DIR, "j0nasw_ai_dictionary.csv")
FINAL_CSV_PATH = os.path.join(BASE_DIR, "ai_dictionary_dataset.csv")


def clean_html_text(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace into clean plain text."""
    text = re.sub(r"<script.*?>.*?</script>", " ", raw_html, flags=re.S | re.I)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_google_ml_glossary() -> list[dict]:
    """
    Scrapes/downloads the official Google Machine Learning Glossary
    (https://developers.google.com/machine-learning/glossary) containing 400+ AI/ML terms and definitions.
    """
    url = "https://developers.google.com/machine-learning/glossary"
    print(f"1. Downloading Official Google Machine Learning Glossary from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        page_html = resp.read().decode("utf-8", errors="ignore")

    # Match <h2 class="hide-from-toc" id="...">Term</h2> followed by definition paragraphs
    pattern = re.compile(
        r'<h2[^>]*class="[^"]*hide-from-toc[^"]*"[^>]*>(.*?)</h2>(.*?)(?=<h2[^>]*class="[^"]*hide-from-toc|<h1|$)',
        re.S | re.I,
    )
    entries = []
    for match in pattern.finditer(page_html):
        raw_term = clean_html_text(match.group(1))
        raw_body = match.group(2)

        # Extract paragraphs from the body
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", raw_body, re.S | re.I)
        cleaned_paras = []
        for p in paragraphs:
            cp = clean_html_text(p)
            # Skip cross-reference/metadata lines like "See ..." if we already have a main definition
            if cp and not cp.lower().startswith(("see the ", "contrast with ", "for more information")):
                cleaned_paras.append(cp)

        if not raw_term or not cleaned_paras:
            continue

        # Take the first 1-2 concise paragraphs as the simple definition
        simple_def = cleaned_paras[0]
        extra_note = cleaned_paras[1] if len(cleaned_paras) > 1 else ""

        entries.append({
            "term": raw_term,
            "simple_definition": simple_def,
            "extra_context": extra_note,
            "source": "Google Machine Learning Glossary (developers.google.com)",
        })

    print(f"   Extracted {len(entries)} official AI/ML terms & definitions from Google ML Glossary!")
    return entries


def infer_category(term: str, definition: str) -> str:
    """Categorize an AI/ML term based on its domain keywords."""
    text = f"{term} {definition}".lower()
    if any(k in text for k in ["transformer", "token", "language model", "llm", "nlp", "word", "text", "bert", "gpt", "translation", "prompt"]):
        return "Natural Language Processing & LLMs"
    if any(k in text for k in ["image", "pixel", "convolutional", "cnn", "bounding box", "vision", "object detection"]):
        return "Computer Vision"
    if any(k in text for k in ["reinforcement", "reward", "agent", "policy", "markov", "q-learning", "action"]):
        return "Reinforcement Learning"
    if any(k in text for k in ["neural network", "layer", "neuron", "backpropagation", "activation", "relu", "sigmoid", "hidden layer", "deep"]):
        return "Deep Learning & Neural Networks"
    if any(k in text for k in ["gradient", "loss", "optimizer", "learning rate", "converge", "regularization", "epoch", "batch"]):
        return "Optimization & Training"
    if any(k in text for k in ["precision", "recall", "accuracy", "roc", "auc", "confusion matrix", "f1", "error", "metric", "validation"]):
        return "Evaluation & Metrics"
    if any(k in text for k in ["feature", "dataset", "missing", "outlier", "normalization", "scaling", "imbalance", "sampling"]):
        return "Data Science & Feature Engineering"
    if any(k in text for k in ["fairness", "bias", "ethics", "safety", "explainab", "privacy", "hallucination"]):
        return "AI Ethics, Safety & Explainability"
    return "Core Machine Learning"


def match_j0nasw_keywords(terms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-references our definitions dataset with the 16,275 real AI keywords from
    Hugging Face's J0nasW/AI-Dictionary (j0nasw_ai_dictionary.csv) to attach real-world
    keyword variants and noisy spellings (like 'graph neuralnetworks', 'back propagation').
    """
    if not os.path.exists(J0NASW_CSV_PATH):
        print("   Note: j0nasw_ai_dictionary.csv not found locally; skipping J0nasW keyword linking.")
        return terms_df

    print("2. Linking terms with Hugging Face J0nasW/AI-Dictionary (16,275 AI keyphrases)...")
    j_df = pd.read_csv(J0NASW_CSV_PATH, usecols=["keyword", "weight"])
    j_df["keyword"] = j_df["keyword"].astype(str).str.strip()
    j_keywords = j_df["keyword"].tolist()

    # Build fast lookup by compact alphanumeric string and root words
    compact_to_variants = {}
    for kw in j_keywords:
        comp = re.sub(r"[^a-z0-9]", "", kw.lower())
        if comp:
            compact_to_variants.setdefault(comp, []).append(kw)

    enriched_aliases = []
    matched_count = 0
    for _, row in terms_df.iterrows():
        term = str(row["term"])
        clean_t = re.sub(r"\s*\([^)]*\)", "", term).strip().lower()
        acronyms = [a.lower() for a in re.findall(r"\(([^)]+)\)", term)]
        comp_t = re.sub(r"[^a-z0-9]", "", clean_t)

        aliases_set = set()
        if str(row.get("aliases", "")).strip() and str(row.get("aliases", "")) != "nan":
            for a in str(row["aliases"]).split(","):
                if a.strip():
                    aliases_set.add(a.strip())

        # Add spaced / unspaced / acronym variations
        aliases_set.add(clean_t)
        for acr in acronyms:
            aliases_set.add(acr)
        if " " not in clean_t and len(clean_t) > 8:
            # e.g., backpropagation -> back propagation
            for prefix in ["back", "over", "under", "hyper", "multi", "auto", "down", "up", "cross", "self", "meta", "fine", "zero", "few"]:
                if clean_t.startswith(prefix) and len(clean_t) > len(prefix) + 3:
                    aliases_set.add(f"{prefix} {clean_t[len(prefix):]}")

        # Match from J0nasW/AI-Dictionary
        j_matches = compact_to_variants.get(comp_t, [])
        if j_matches:
            matched_count += 1
            for m in j_matches[:5]:
                aliases_set.add(m)

        enriched_aliases.append(", ".join(sorted(aliases_set)))

    terms_df["aliases"] = enriched_aliases
    print(f"   Matched {matched_count} terms directly with J0nasW/AI-Dictionary keyphrase variations!")
    return terms_df


def build_combined_real_dataset():
    # 1. Fetch official Google Machine Learning Glossary (400+ real terms & definitions)
    google_entries = fetch_google_ml_glossary()

    # 2. Load any existing curated student analogies if present to keep extra analogies
    existing_map = {}
    if os.path.exists(FINAL_CSV_PATH):
        try:
            old_df = pd.read_csv(FINAL_CSV_PATH)
            for _, r in old_df.iterrows():
                key = re.sub(r"\s*\([^)]*\)", "", str(r["term"])).strip().lower()
                existing_map[key] = r.to_dict()
        except Exception:
            pass

    rows = []
    seen_keys = set()

    for item in google_entries:
        term = item["term"]
        key = re.sub(r"\s*\([^)]*\)", "", term).strip().lower()
        if key in seen_keys or len(key) < 2:
            continue
        seen_keys.add(key)

        simple_def = item["simple_definition"]
        extra_context = item["extra_context"]
        category = infer_category(term, f"{simple_def} {extra_context}")

        # If we also have a student analogy or extra aliases, preserve them
        prev = existing_map.get(key, {})
        analogy = prev.get("analogy", "") if isinstance(prev.get("analogy", ""), str) else ""
        if not analogy:
            analogy = extra_context if extra_context else f"In {category}, {term} is a core concept used when building or evaluating AI systems."

        rows.append({
            "term": term,
            "category": category,
            "aliases": prev.get("aliases", key),
            "simple_definition": simple_def,
            "analogy": analogy,
            "source": item["source"],
        })

    # Also keep any terms from our starter list that weren't in the scraped page
    for key, prev in existing_map.items():
        if key not in seen_keys:
            seen_keys.add(key)
            rows.append({
                "term": prev["term"],
                "category": prev["category"],
                "aliases": prev.get("aliases", ""),
                "simple_definition": prev["simple_definition"],
                "analogy": prev.get("analogy", ""),
                "source": "Google ML Glossary + J0nasW/AI-Dictionary",
            })

    df = pd.DataFrame(rows)
    df = match_j0nasw_keywords(df)
    df = df.sort_values("term").reset_index(drop=True)
    df.to_csv(FINAL_CSV_PATH, index=False, encoding="utf-8")
    print(f"3. Saved combined real AI Dictionary dataset ({len(df)} terms) to {FINAL_CSV_PATH}")
    return df


if __name__ == "__main__":
    build_combined_real_dataset()
    from train import train_and_save
    train_and_save()
