"""
Node (user) feature engineering.

Features:
  degree_norm     — log(1 + degree)
  out_degree_norm — log(1 + out_degree)
  in_degree_norm  — log(1 + in_degree)
  is_external     — user present in events but not in users table

Produces a node feature matrix indexed by user idx.

Run: python learner/data/features/node.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from learner.config import PROC_DIR, RAW_DIR


def build_node_features(users_df: pd.DataFrame, user2idx: dict,
                        external_users: set) -> pd.DataFrame:
    """
    Build node feature DataFrame for all users.
    Returns DataFrame with columns: id, idx, degree_norm, out_degree_norm,
    in_degree_norm, is_external.
    """
    users_df = users_df.copy()
    users_df["degree_norm"]     = np.log1p(users_df["degree"].fillna(0))
    users_df["out_degree_norm"] = np.log1p(users_df["out_degree"].fillna(0))
    users_df["in_degree_norm"]  = np.log1p(users_df["in_degree"].fillna(0))
    users_df["is_external"]     = ~users_df["id"].astype(str).isin(
        set(user2idx.keys()) - external_users
    )

    # Add external users with zero features
    if external_users:
        ext_df = pd.DataFrame({
            "id":             list(external_users),
            "out_degree":     0,
            "in_degree":      0,
            "degree":         0,
            "degree_norm":    0.0,
            "out_degree_norm": 0.0,
            "in_degree_norm": 0.0,
            "is_external":    True,
        })
        users_df = pd.concat([users_df, ext_df], ignore_index=True)

    users_df["idx"] = users_df["id"].astype(str).map(user2idx)
    return users_df


def build_node_feature_matrix(node_feats_df: pd.DataFrame,
                               feat_cols: list = None) -> np.ndarray:
    """
    Build a dense [max_idx+1, len(feat_cols)] float32 matrix.
    Rows not in node_feats_df are zero.
    """
    if feat_cols is None:
        feat_cols = ["degree_norm", "out_degree_norm", "in_degree_norm"]

    df = node_feats_df.dropna(subset=["idx"]).copy()
    df["idx"] = df["idx"].astype(int)
    max_idx = int(df["idx"].max()) + 1
    matrix = np.zeros((max_idx, len(feat_cols)), dtype=np.float32)
    idx_arr = df["idx"].values
    for i, col in enumerate(feat_cols):
        matrix[idx_arr, i] = df[col].fillna(0).values
    return matrix


if __name__ == "__main__":
    print("=== features/node.py ===\n")
    users_df = pd.read_parquet(RAW_DIR / "users.parquet")
    id_map   = pd.read_parquet(PROC_DIR / "id_map.parquet")

    user_map = id_map[id_map["node_type"] == "user"]
    user2idx = dict(zip(user_map["str_id"].astype(str), user_map["idx"].astype(int)))
    external_users = set(user_map[user_map["is_external"]]["str_id"].astype(str))

    node_feats = build_node_features(users_df, user2idx, external_users)
    node_feats.to_parquet(PROC_DIR / "node_features.parquet", index=False)
    print(f"✅ node_features: {len(node_feats):,} users → {PROC_DIR / 'node_features.parquet'}")
