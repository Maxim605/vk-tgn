"""
Text feature engineering: BERT embeddings with disk cache.

Encodes event texts using ruBERT (DeepPavlov/rubert-base-cased).
Empty strings → zero vector (not passed through BERT).
Cache stored as compressed numpy archive to avoid re-encoding.

Run: python learner/data/features/text.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from learner.config import (
    PROC_DIR, MODEL_CACHE, BERT_MODEL, BERT_DIM, BERT_BATCH, BERT_MAXLEN,
)

CACHE_FILE = PROC_DIR / "text_emb_cache.npz"


def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("  Intel Arc GPU (XPU)")
        return torch.device("xpu")
    elif torch.cuda.is_available():
        print("  CUDA GPU")
        return torch.device("cuda")
    print("  CPU")
    return torch.device("cpu")


def load_bert(device: torch.device):
    from transformers import AutoTokenizer, AutoModel

    local_path = MODEL_CACHE / "rubert-base-cased"
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"  BERT from cache: {local_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(local_path))
        model = AutoModel.from_pretrained(str(local_path)).eval().to(device)
    else:
        print(f"  Downloading BERT: {BERT_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model = AutoModel.from_pretrained(BERT_MODEL).eval().to(device)
        tokenizer.save_pretrained(str(local_path))
        model.save_pretrained(str(local_path))
        print(f"  Saved → {local_path}")
    return tokenizer, model


@torch.no_grad()
def encode_unique(unique_texts: list, tokenizer, model,
                  device: torch.device) -> np.ndarray:
    """Encode a list of unique texts → float32 [N, BERT_DIM]."""
    results = []
    n = len(unique_texts)
    for i in range(0, n, BERT_BATCH):
        batch = unique_texts[i : i + BERT_BATCH]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=BERT_MAXLEN, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        results.append(cls)
        done = min(i + BERT_BATCH, n)
        print(f"    BERT: {done:,}/{n:,}  ({done/n*100:.1f}%)", end="\r")
    print()
    return np.vstack(results)


def get_embeddings(texts: list, device: torch.device = None) -> np.ndarray:
    """
    Return embeddings [N, BERT_DIM] for a list of texts.
    Uses disk cache; only encodes texts not already cached.
    Empty strings → zero vector.
    """
    if device is None:
        device = get_device()

    EMPTY_EMB = np.zeros(BERT_DIM, dtype=np.float32)

    # Load cache
    cached_texts: list = []
    cached_embs: np.ndarray = np.empty((0, BERT_DIM), dtype=np.float32)
    if CACHE_FILE.exists():
        data = np.load(CACHE_FILE, allow_pickle=True)
        cached_texts = data["texts"].tolist()
        cached_embs  = data["embs"]
        print(f"  Cache: {len(cached_texts):,} texts")

    cached_map = {t: i for i, t in enumerate(cached_texts)}

    new_texts = sorted({t for t in texts if t and t not in cached_map})
    print(f"  New texts for BERT: {len(new_texts):,}")

    if new_texts:
        tokenizer, model = load_bert(device)
        new_embs = encode_unique(new_texts, tokenizer, model, device)

        all_texts = cached_texts + new_texts
        all_embs  = (
            np.vstack([cached_embs, new_embs]) if len(cached_embs) else new_embs
        )
        np.savez_compressed(CACHE_FILE, texts=np.array(all_texts), embs=all_embs)
        print(f"  Cache saved: {len(all_texts):,} texts → {CACHE_FILE}")

        cached_texts = all_texts
        cached_embs  = all_embs
        cached_map   = {t: i for i, t in enumerate(cached_texts)}

    result = np.empty((len(texts), BERT_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        result[i] = cached_embs[cached_map[t]] if (t and t in cached_map) else EMPTY_EMB
    return result


if __name__ == "__main__":
    import pandas as pd
    print("=== features/text.py ===\n")
    events_df = pd.read_parquet(PROC_DIR / "canonical_events.parquet")
    texts = events_df["text"].fillna("").tolist()
    n_empty = sum(1 for t in texts if not t)
    unique_nonempty = len({t for t in texts if t})
    print(f"  Empty: {n_empty:,}  Unique non-empty: {unique_nonempty:,}")
    device = get_device()
    embs = get_embeddings(texts, device)
    print(f"✅ text_embs: {embs.shape}")
