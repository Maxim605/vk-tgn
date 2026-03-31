"""
Data integrity checks called from dataset_builder.py.

All functions raise AssertionError with a descriptive message on failure.
"""

import torch
from torch_geometric.data import TemporalData


def check_temporal_order(data: TemporalData) -> None:
    """Events must be in non-decreasing time order."""
    t = data.t
    assert (t[1:] >= t[:-1]).all(), \
        f"Temporal order violated: found {(t[1:] < t[:-1]).sum()} out-of-order events"


def check_no_future_leakage(train: TemporalData, val: TemporalData,
                             test: TemporalData) -> None:
    """Train max-t ≤ val min-t ≤ test min-t."""
    assert float(train.t.max()) <= float(val.t.min()), \
        f"Leakage: train.t.max={train.t.max():.0f} > val.t.min={val.t.min():.0f}"
    assert float(val.t.max()) <= float(test.t.min()), \
        f"Leakage: val.t.max={val.t.max():.0f} > test.t.min={test.t.min():.0f}"


def check_no_negative_ids(data: TemporalData) -> None:
    """All src and dst node IDs must be non-negative."""
    assert (data.src >= 0).all(), \
        f"Negative src IDs: {(data.src < 0).sum()} events"
    assert (data.dst >= 0).all(), \
        f"Negative dst IDs: {(data.dst < 0).sum()} events"


def check_closed_world(data: TemporalData, valid_ids: set) -> None:
    """All dst IDs must be in the closed-world valid_ids set."""
    dst_set = set(data.dst.numpy().tolist())
    outside = dst_set - valid_ids
    assert len(outside) == 0, \
        f"Closed-world violation: {len(outside)} dst IDs not in valid_ids"


def check_msg_dim(data: TemporalData, expected_dim: int) -> None:
    """Message tensor must have the expected feature dimension."""
    actual = data.msg.shape[1]
    assert actual == expected_dim, \
        f"msg dim mismatch: expected {expected_dim}, got {actual}"


def run_all_checks(train: TemporalData, val: TemporalData,
                   test: TemporalData) -> None:
    """Run all standard checks on train/val/test splits."""
    for name, split in [("train", train), ("val", val), ("test", test)]:
        check_temporal_order(split)
        check_no_negative_ids(split)
        print(f"  ✅ {name}: temporal_order OK, no_negative_ids OK")
    check_no_future_leakage(train, val, test)
    print("  ✅ no_future_leakage OK")
