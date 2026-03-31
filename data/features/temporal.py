"""
Temporal feature engineering for the event stream.

Features computed causally (no future leakage):
  delta_t_user        — time since user's last event (groupby+diff on sorted stream)
  delta_t_post        — time since post's last event
  post_age            — t - post.created_at
  post_comments_so_far — cumcount of comment events on this post up to (not including) current

Normalization:
  Uses 75th percentile from train split (not median, which can be 0).
  Formula: log(1 + x / (p75 + 1e-6))

Run: python learner/data/features/temporal.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from learner.config import PROC_DIR, RAW_DIR, TRAIN_FRAC


def compute_temporal_features(events_df: pd.DataFrame, posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to events_df (must be sorted by t).
    Returns events_df with new columns added in-place.
    """
    assert (events_df["t"].diff().dropna() >= 0).all(), \
        "events_df must be sorted by t before computing temporal features"

    # delta_t_user: time since this user's previous event
    events_df["delta_t_user"] = (
        events_df.groupby("src_idx")["t"].diff().fillna(0)
    )

    # delta_t_post: time since this post's previous event
    events_df["delta_t_post"] = (
        events_df.groupby("dst_idx")["t"].diff().fillna(0)
    )

    # post_age: t - post.created_at (clipped to >= 0)
    post_date = posts_df.set_index("id")["date"].to_dict()
    events_df["post_created_at"] = events_df["dst"].astype(str).map(post_date)
    events_df["post_age"] = (
        events_df["t"] - pd.to_numeric(events_df["post_created_at"], errors="coerce")
    ).clip(lower=0).fillna(0)

    # post_comments_so_far: causal cumcount of comment events per post
    comment_mask = events_df["type"] == "comment"
    events_df["post_comments_so_far"] = 0.0
    events_df.loc[comment_mask, "post_comments_so_far"] = (
        events_df[comment_mask].groupby("dst_idx").cumcount().astype(float)
    )

    # type one-hot
    events_df["type_write"]   = (events_df["type"] == "write").astype(float)
    events_df["type_comment"] = (events_df["type"] == "comment").astype(float)

    return events_df


def normalize_temporal(events_df: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """
    Normalize temporal features using 75th percentile from train split.
    Uses p75 instead of median to avoid division by zero when median=0.
    """
    def norm(x, p75):
        return np.log1p(x / (p75 + 1e-6))

    train = events_df.iloc[:train_end]

    p75_delta_user    = float(np.percentile(train["delta_t_user"],    75))
    p75_delta_post    = float(np.percentile(train["delta_t_post"],    75))
    p75_post_age      = float(np.percentile(train["post_age"],        75))
    p75_post_comments = float(np.percentile(train["post_comments_so_far"], 75))

    print(f"  Train p75 — delta_t_user: {p75_delta_user:.0f}s  "
          f"delta_t_post: {p75_delta_post:.0f}s  "
          f"post_age: {p75_post_age/86400:.1f}d  "
          f"post_comments: {p75_post_comments:.1f}")

    events_df["delta_t_user_norm"]    = norm(events_df["delta_t_user"],    p75_delta_user)
    events_df["delta_t_post_norm"]    = norm(events_df["delta_t_post"],    p75_delta_post)
    events_df["post_age_norm"]        = norm(events_df["post_age"],        p75_post_age)
    events_df["post_comments_norm"]   = norm(events_df["post_comments_so_far"], p75_post_comments)
    events_df["post_likes_norm"]      = 0.0   # likes have no timestamp → zero to avoid leakage

    return events_df


if __name__ == "__main__":
    print("=== features/temporal.py ===\n")
    events_df = pd.read_parquet(PROC_DIR / "canonical_events.parquet")
    posts_df  = pd.read_parquet(RAW_DIR / "posts.parquet")

    events_df = events_df.sort_values("t").reset_index(drop=True)
    train_end = int(len(events_df) * TRAIN_FRAC)

    events_df = compute_temporal_features(events_df, posts_df)
    events_df = normalize_temporal(events_df, train_end)

    print(f"Temporal features added: {len(events_df):,} events")
    events_df.to_parquet(PROC_DIR / "events_temporal.parquet", index=False)
    print(f"✅ Saved → {PROC_DIR / 'events_temporal.parquet'}")
