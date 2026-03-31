"""
Post (static) feature engineering.

Features:
  likes_norm    — set to 0 (likes have no timestamp → would cause leakage)
  text_len_norm — log(1 + len(text))
  has_attach    — boolean attachment flag

Run: python learner/data/features/post.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from learner.config import PROC_DIR, RAW_DIR


def build_post_features(posts_df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized static features to posts_df."""
    posts_df = posts_df.copy()
    posts_df["likes_norm"]    = 0.0   # no timestamp → zero to avoid leakage
    posts_df["text_len"]      = posts_df["text"].str.len().fillna(0)
    posts_df["text_len_norm"] = np.log1p(posts_df["text_len"])
    posts_df = posts_df.drop_duplicates(subset=["id"], keep="first")
    return posts_df


if __name__ == "__main__":
    print("=== features/post.py ===\n")
    posts_df = pd.read_parquet(RAW_DIR / "posts.parquet")
    posts_df = build_post_features(posts_df)
    posts_df.to_parquet(PROC_DIR / "post_features.parquet", index=False)
    print(f"✅ post_features: {len(posts_df):,} posts → {PROC_DIR / 'post_features.parquet'}")
