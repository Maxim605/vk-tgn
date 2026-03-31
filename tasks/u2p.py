"""
User → Post link prediction task.

Filters the canonical event stream to fof users and builds
TemporalData splits for the u2p task.

neg_pool: all unique dst (post) IDs seen in the canonical stream.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import TemporalData

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from learner.config import PROC_DIR


def build_u2p_splits(saved: dict, fof_idx: set, msg_cols: list = None):
    """
    Filter saved TemporalData to fof users and return u2p splits.

    Args:
        saved:    dict with keys full/train/val/test/split_info
        fof_idx:  set of canonical user idx values in the fof subgraph
        msg_cols: list of column indices to keep from msg (None = keep all)

    Returns:
        train, val, test, neg_pool (all on CPU)
    """
    full: TemporalData = saved["full"]
    src_np = full.src.numpy()
    mask = torch.tensor([int(s) in fof_idx for s in src_np], dtype=torch.bool)

    def _filt(data: TemporalData) -> TemporalData:
        m = torch.tensor(
            [int(s) in fof_idx for s in data.src.numpy()], dtype=torch.bool
        )
        msg = data.msg[m] if msg_cols is None else data.msg[m][:, msg_cols]
        return TemporalData(
            src=data.src[m],
            dst=data.dst[m],
            t=data.t[m].float(),
            msg=msg,
        )

    train = _filt(saved["train"])
    val   = _filt(saved["val"])
    test  = _filt(saved["test"])

    # neg_pool: all unique post dst IDs in the full filtered stream
    full_filt = _filt(full)
    neg_pool  = full_filt.dst.unique()

    return train, val, test, neg_pool
