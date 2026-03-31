"""
Temporal train/val/test split computation.

Splits are computed by time quantile (not row index) so that all events
with the same timestamp stay in the same split — no boundary leakage.

Outputs (learner/data/processed/):
  split_info.json — split boundary indices and timestamps

Run: python learner/data/splits.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from learner.config import PROC_DIR, TRAIN_FRAC, VAL_FRAC


def compute_splits(events_df: pd.DataFrame) -> dict:
    """
    Compute temporal split boundaries.
    Returns dict with train_end, val_end, timestamps, and counts.
    """
    t_sorted = np.sort(events_df["t"].values)
    t_train  = float(np.quantile(t_sorted, TRAIN_FRAC))
    t_val    = float(np.quantile(t_sorted, VAL_FRAC))

    # Shift boundaries to end of same-timestamp group
    train_end = int((events_df["t"] <= t_train).sum())
    val_end   = int((events_df["t"] <= t_val).sum())
    n         = len(events_df)

    split_info = {
        "n_events":    n,
        "train_end":   train_end,
        "val_end":     val_end,
        "test_end":    n,
        "train_t_max": float(events_df.iloc[train_end - 1]["t"]),
        "val_t_max":   float(events_df.iloc[val_end - 1]["t"]),
        "test_t_max":  float(events_df.iloc[-1]["t"]),
    }
    return split_info


if __name__ == "__main__":
    import pandas as pd
    print("=== splits.py ===\n")
    events_df = pd.read_parquet(PROC_DIR / "canonical_events.parquet")
    events_df = events_df.sort_values("t").reset_index(drop=True)

    split_info = compute_splits(events_df)

    print(f"Temporal split:")
    print(f"  train: 0..{split_info['train_end']:,}  "
          f"t_max={pd.to_datetime(split_info['train_t_max'], unit='s')}")
    print(f"  val:   {split_info['train_end']:,}..{split_info['val_end']:,}  "
          f"t_max={pd.to_datetime(split_info['val_t_max'], unit='s')}")
    print(f"  test:  {split_info['val_end']:,}..{split_info['n_events']:,}  "
          f"t_max={pd.to_datetime(split_info['test_t_max'], unit='s')}")

    out = PROC_DIR / "split_info.json"
    out.write_text(json.dumps(split_info, indent=2))
    print(f"\n✅ Saved → {out}")
