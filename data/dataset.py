"""
Этап 1.4 + 2: BERT-энкодинг текстов и сборка TemporalData.
BERT-эмбеддинги кешируются в learner/data/processed/text_emb_cache.npz
Запуск: python learner/data/dataset.py
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import TemporalData

PROC_DIR    = Path(__file__).parent / "processed"
CACHE_FILE  = PROC_DIR / "text_emb_cache.npz"   # сжатый numpy архив
OUT_FILE    = PROC_DIR / "temporal_data.pt"
MODEL_CACHE = Path(__file__).parent / "model_cache"

# ── Устройство ────────────────────────────────────────────────────────────────
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("  Intel Arc GPU (XPU)")
        return torch.device("xpu")
    elif torch.cuda.is_available():
        print("  CUDA GPU")
        return torch.device("cuda")
    print("  CPU")
    return torch.device("cpu")

# ── BERT ──────────────────────────────────────────────────────────────────────
def load_bert(device):
    from transformers import AutoTokenizer, AutoModel
    local_path = MODEL_CACHE / "rubert-base-cased"
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"  BERT из кеша: {local_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(local_path))
        model = AutoModel.from_pretrained(str(local_path)).eval().to(device)
    else:
        name = "DeepPavlov/rubert-base-cased"
        print(f"  Загрузка BERT: {name}")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval().to(device)
        tokenizer.save_pretrained(str(local_path))
        model.save_pretrained(str(local_path))
        print(f"  Сохранено → {local_path}")
    return tokenizer, model


@torch.no_grad()
def encode_unique(unique_texts: list, tokenizer, model, device,
                  batch_size: int = 256, max_length: int = 128) -> np.ndarray:
    """Кодирует список уникальных текстов → float32 [N, 768]."""
    results = []
    n = len(unique_texts)
    for i in range(0, n, batch_size):
        batch = unique_texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        results.append(cls)
        done = min(i + batch_size, n)
        print(f"    BERT: {done:,}/{n:,}  ({done/n*100:.1f}%)", end="\r")
    print()
    return np.vstack(results)


def get_embeddings(texts: list, device) -> np.ndarray:
    """
    Возвращает эмбеддинги [N, 768] для списка текстов.
    Кеш: уникальные тексты → индекс → .npz файл.
    Пустые строки → нулевой вектор (не гоним через BERT).
    """
    EMPTY_EMB = np.zeros(768, dtype=np.float32)

    # Загружаем кеш
    cached_texts: list = []
    cached_embs:  np.ndarray = np.empty((0, 768), dtype=np.float32)
    if CACHE_FILE.exists():
        data = np.load(CACHE_FILE, allow_pickle=True)
        cached_texts = data["texts"].tolist()
        cached_embs  = data["embs"]
        print(f"  Кеш: {len(cached_texts):,} текстов")

    cached_map = {t: i for i, t in enumerate(cached_texts)}

    # Уникальные непустые тексты, которых нет в кеше
    new_texts = sorted({t for t in texts if t and t not in cached_map})
    print(f"  Новых текстов для BERT: {len(new_texts):,}")

    if new_texts:
        tokenizer, model = load_bert(device)
        new_embs = encode_unique(new_texts, tokenizer, model, device)

        # Объединяем с кешем
        all_texts = cached_texts + new_texts
        all_embs  = np.vstack([cached_embs, new_embs]) if len(cached_embs) else new_embs

        np.savez_compressed(CACHE_FILE, texts=np.array(all_texts), embs=all_embs)
        print(f"  Кеш сохранён: {len(all_texts):,} текстов → {CACHE_FILE}")

        cached_texts = all_texts
        cached_embs  = all_embs
        cached_map   = {t: i for i, t in enumerate(cached_texts)}

    # Собираем результат в порядке входного списка
    result = np.empty((len(texts), 768), dtype=np.float32)
    for i, t in enumerate(texts):
        if t and t in cached_map:
            result[i] = cached_embs[cached_map[t]]
        else:
            result[i] = EMPTY_EMB
    return result


# ── Основная функция ──────────────────────────────────────────────────────────
def build_temporal_data():
    print("=== dataset.py ===\n")
    print("Загрузка events_featured...")
    events_df = pd.read_parquet(PROC_DIR / "events_featured.parquet")
    split_info = json.loads((PROC_DIR / "split_info.json").read_text())
    print(f"  Событий: {len(events_df):,}")

    device = get_device()

    # ── BERT ──────────────────────────────────────────────────────────────────
    print("\nBERT энкодинг...")
    texts = events_df["text"].fillna("").tolist()

    # Статистика текстов
    n_empty = sum(1 for t in texts if not t)
    unique_nonempty = len({t for t in texts if t})
    print(f"  Пустых: {n_empty:,}  Уникальных непустых: {unique_nonempty:,}")

    text_embs = get_embeddings(texts, device)  # [N, 768]
    print(f"  text_embs: {text_embs.shape}")

    # ── msg тензор ────────────────────────────────────────────────────────────
    print("\nСборка msg...")

    # Node features пользователей (src) — векторизованно, без iterrows
    node_feats_df = pd.read_parquet(PROC_DIR / "node_features.parquet")
    node_feats_df = node_feats_df.dropna(subset=["idx"]).copy()
    node_feats_df["idx"] = node_feats_df["idx"].astype(int)
    feat_cols = ["degree_norm", "out_degree_norm", "in_degree_norm"]
    max_idx = int(node_feats_df["idx"].max()) + 1
    node_feat_matrix = np.zeros((max_idx, len(feat_cols)), dtype=np.float32)
    idx_arr = node_feats_df["idx"].values
    for i, col in enumerate(feat_cols):
        node_feat_matrix[idx_arr, i] = node_feats_df[col].fillna(0).values
    src_node_feats = node_feat_matrix[events_df["src_idx"].values]  # [N, 3]

    temporal = events_df[[
        "type_write", "type_comment",
        "delta_t_user_norm", "delta_t_post_norm", "post_age_norm",
        "post_likes_norm", "post_comments_norm",
    ]].values.astype(np.float32)  # [N, 7]

    # msg = [BERT(768) | temporal(7) | node_feats(3)] = 778 dims
    msg = np.concatenate([text_embs, temporal, src_node_feats], axis=1)
    print(f"  msg: {msg.shape}  (768 BERT + 7 temporal + 3 node_feats)")

    # ── TemporalData ──────────────────────────────────────────────────────────
    print("\nСборка TemporalData...")
    data = TemporalData(
        src  = torch.tensor(events_df["src_idx"].values, dtype=torch.long),
        dst  = torch.tensor(events_df["dst_idx"].values, dtype=torch.long),
        t    = torch.tensor(events_df["t"].values,       dtype=torch.float64),
        msg  = torch.tensor(msg,                         dtype=torch.float32),
        y    = torch.tensor(
            (events_df["type"] == "write").astype(int).values, dtype=torch.long
        ),
    )
    # dst_user: индекс владельца поста (для user→user задачи)
    # NaN → -1 (внешний или неизвестный owner)
    dst_user = events_df["dst_user_idx"].fillna(-1).astype(int).values
    data.dst_user = torch.tensor(dst_user, dtype=torch.long)
    print(f"  num_events: {data.num_events:,}")
    print(f"  num_nodes:  {data.num_nodes:,}")

    # ── Temporal split ────────────────────────────────────────────────────────
    n         = data.num_events
    train_end = split_info["train_end"]
    val_end   = split_info["val_end"]

    train_data = data[:train_end]
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:]

    print(f"\nSplit: train={train_data.num_events:,}  val={val_data.num_events:,}  test={test_data.num_events:,}")

    # ── Сохранение ────────────────────────────────────────────────────────────
    torch.save({
        "full":       data,
        "train":      train_data,
        "val":        val_data,
        "test":       test_data,
        "split_info": split_info,
    }, OUT_FILE)
    print(f"\n✅ Сохранено → {OUT_FILE}")
    return data, train_data, val_data, test_data


if __name__ == "__main__":
    build_temporal_data()
