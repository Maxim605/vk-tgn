"""
Stage 3: Build TemporalData tensors from processed features.

Assembles the msg tensor:
  msg = [BERT(768) | temporal(7) | node_feats(3)] = 778 dims
  (or msg = [temporal(7) | node_feats(3)] = 10 dims without BERT)

Outputs (learner/data/processed/):
  temporal_data.pt — dict with full/train/val/test TemporalData + split_info

Run: python learner/data/dataset_builder.py [--no-bert]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import TemporalData

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from learner.config import PROC_DIR, RAW_DIR, BERT_DIM
from learner.data.checks import run_all_checks
from learner.data.features.node import build_node_feature_matrix
from learner.data.features.temporal import compute_temporal_features, normalize_temporal
from learner.data.features.text import get_device, get_embeddings
from learner.data.splits import compute_splits


def build(use_bert: bool = False) -> dict:
    print("=== dataset_builder.py ===\n")

    print("Loading data...")
    events_df = pd.read_parquet(PROC_DIR / "canonical_events.parquet")
    posts_df  = pd.read_parquet(RAW_DIR / "posts.parquet")
    print(f"  events: {len(events_df):,}")

    events_df = events_df.sort_values("t").reset_index(drop=True)

    # ── Temporal features ─────────────────────────────────────────────────────
    print("\nComputing temporal features...")
    split_info = compute_splits(events_df)
    train_end  = split_info["train_end"]

    events_df = compute_temporal_features(events_df, posts_df)
    events_df = normalize_temporal(events_df, train_end)

    # ── Node features ─────────────────────────────────────────────────────────
    print("\nLoading node features...")
    node_feats_df = pd.read_parquet(PROC_DIR / "node_features.parquet")
    feat_cols = ["degree_norm", "out_degree_norm", "in_degree_norm"]
    node_feat_matrix = build_node_feature_matrix(node_feats_df, feat_cols)
    src_node_feats = node_feat_matrix[events_df["src_idx"].values]  # [N, 3]

    # ── Temporal msg columns ──────────────────────────────────────────────────
    temporal_cols = [
        "type_write", "type_comment",
        "delta_t_user_norm", "delta_t_post_norm", "post_age_norm",
        "post_likes_norm", "post_comments_norm",
    ]
    temporal = events_df[temporal_cols].values.astype(np.float32)  # [N, 7]

    # ── BERT embeddings ───────────────────────────────────────────────────────
    if use_bert:
        print("\nBERT encoding...")
        texts = events_df["text"].fillna("").tolist()
        n_empty = sum(1 for t in texts if not t)
        unique_nonempty = len({t for t in texts if t})
        print(f"  Empty: {n_empty:,}  Unique non-empty: {unique_nonempty:,}")
        device = get_device()
        text_embs = get_embeddings(texts, device)  # [N, 768]
        print(f"  text_embs: {text_embs.shape}")
        msg = np.concatenate([text_embs, temporal, src_node_feats], axis=1)
        print(f"\nmsg: {msg.shape}  ({BERT_DIM} BERT + 7 temporal + 3 node_feats)")
    else:
        msg = np.concatenate([temporal, src_node_feats], axis=1)
        print(f"\nmsg: {msg.shape}  (7 temporal + 3 node_feats, no BERT)")

    # ── TemporalData ──────────────────────────────────────────────────────────
    print("\nBuilding TemporalData...")
    dst_user = events_df["dst_user_idx"].fillna(-1).astype(int).values

    data = TemporalData(
        src=torch.tensor(events_df["src_idx"].values, dtype=torch.long),
        dst=torch.tensor(events_df["dst_idx"].values, dtype=torch.long),
        t=torch.tensor(events_df["t"].values, dtype=torch.float64),
        msg=torch.tensor(msg, dtype=torch.float32),
        y=torch.tensor(
            (events_df["type"] == "write").astype(int).values, dtype=torch.long
        ),
    )
    data.dst_user = torch.tensor(dst_user, dtype=torch.long)

    # ── Splits ────────────────────────────────────────────────────────────────
    val_end = split_info["val_end"]
    train_data = data[:train_end]
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:]

    print(f"\nSplit: train={train_data.num_events:,}  "
          f"val={val_data.num_events:,}  test={test_data.num_events:,}")

    # ── Integrity checks ──────────────────────────────────────────────────────
    print("\nRunning integrity checks...")
    run_all_checks(train_data, val_data, test_data)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = PROC_DIR / "temporal_data.pt"
    torch.save({
        "full":       data,
        "train":      train_data,
        "val":        val_data,
        "test":       test_data,
        "split_info": split_info,
    }, out)
    (PROC_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2))
    print(f"\n✅ Saved → {out}")

    return {"full": data, "train": train_data, "val": val_data,
            "test": test_data, "split_info": split_info}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bert", action="store_true",
                        help="Skip BERT encoding (faster, smaller msg)")
    args = parser.parse_args()
    build(use_bert=not args.no_bert)
