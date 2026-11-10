import os
import json
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HF_PARQUET_PATH = os.path.join(BASE_DIR, "j0nasw_ai_dictionary.parquet")
HF_CSV_PATH = os.path.join(BASE_DIR, "j0nasw_ai_dictionary.csv")


def download_j0nasw_ai_dictionary():
    """
    Downloads the official J0nasW/AI-Dictionary dataset directly from Hugging Face Hub.
    Repository: https://huggingface.co/datasets/J0nasW/AI-Dictionary
    """
    print("Fetching dataset metadata from Hugging Face Hub (J0nasW/AI-Dictionary)...")
    api_url = "https://huggingface.co/api/datasets/J0nasW/AI-Dictionary"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        meta = json.loads(resp.read().decode("utf-8"))

    siblings = [s["rfilename"] for s in meta.get("siblings", [])]
    print("Files inside J0nasW/AI-Dictionary repo:", siblings)

    # 1. Try downloading raw CSV / Parquet / JSONL file directly from the repo
    data_files = [
        f for f in siblings
        if f.endswith((".parquet", ".csv", ".json", ".jsonl", ".tsv", ".xlsx"))
    ]

    if data_files:
        target_file = data_files[0]
        download_url = f"https://huggingface.co/datasets/J0nasW/AI-Dictionary/resolve/main/{urllib.request.quote(target_file)}"
        local_file = os.path.join(BASE_DIR, os.path.basename(target_file))
        print(f"Downloading {download_url} -> {local_file} ...")
        urllib.request.urlretrieve(download_url, local_file)

        if target_file.endswith(".parquet"):
            df = pd.read_parquet(local_file)
        elif target_file.endswith(".csv"):
            df = pd.read_csv(local_file)
        elif target_file.endswith(".tsv"):
            df = pd.read_csv(local_file, sep="\t")
        elif target_file.endswith(".jsonl"):
            df = pd.read_json(local_file, lines=True)
        elif target_file.endswith(".json"):
            df = pd.read_json(local_file)
        else:
            df = pd.read_excel(local_file)
    else:
        # 2. Fallback to Hugging Face auto-converted Parquet endpoint
        parquet_api = "https://huggingface.co/api/datasets/J0nasW/AI-Dictionary/parquet/default/train"
        req2 = urllib.request.Request(parquet_api, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=30) as resp2:
            parquet_urls = json.loads(resp2.read().decode("utf-8"))
        print("Downloading Hugging Face Parquet shard:", parquet_urls[0])
        urllib.request.urlretrieve(parquet_urls[0], HF_PARQUET_PATH)
        df = pd.read_parquet(HF_PARQUET_PATH)

    print(f"Successfully downloaded J0nasW/AI-Dictionary! Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head(5))

    df.to_csv(HF_CSV_PATH, index=False, encoding="utf-8")
    print(f"\nSaved official dataset to: {HF_CSV_PATH}")
    return df


if __name__ == "__main__":
    download_j0nasw_ai_dictionary()
