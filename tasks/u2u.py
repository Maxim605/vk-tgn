"""
User → User link prediction task (closed world).

Closed-world definition:
  - src must be in fof_idx
  - dst (post_owner) must ALSO be in fof_idx

neg_pool: ALL users in the closed-world fof set (not just seen dst).
This is stricter than u2p and tests whether the model can distinguish
which fof user a given user will interact with.

Link predictor input:
  concat([z_src, z_dst, z_src * z_dst, (z_src - z_dst).abs()])
  — asymmetric, captures directed interaction signal.
"""

import sys
from pathlib import Path

import torch
from torch_geometric.data import TemporalData

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from learner.config import PROC_DIR


def build_u2u_splits(saved: dict, fof_idx: set, msg_cols: list = None):
    """
    Filter saved TemporalData to closed-world u2u events and return splits.

    Closed world: both src AND dst (post_owner) must be in fof_idx.

    Args:
        saved:    dict with keys full/train/val/test/split_info
        fof_idx:  set of canonical user idx values in the fof subgraph
        msg_cols: list of column indices to keep from msg (None = keep all)

    Returns:
        train, val, test, neg_pool (all on CPU)
        neg_pool contains ALL fof user indices (closed-world negatives)
    """

    def _filt(data: TemporalData) -> TemporalData:
        if not hasattr(data, "dst_user"):
            raise ValueError("TemporalData must have dst_user attribute for u2u task")

        src_np      = data.src.numpy()
        dst_user_np = data.dst_user.numpy()

        # Closed world: both src and post_owner must be in fof
        mask = torch.tensor(
            [
                int(s) in fof_idx and int(du) in fof_idx
                for s, du in zip(src_np, dst_user_np)
            ],
            dtype=torch.bool,
        )
        # Also filter out events with no valid owner (dst_user == -1)
        valid_owner = data.dst_user >= 0
        mask = mask & valid_owner

        msg = data.msg[mask] if msg_cols is None else data.msg[mask][:, msg_cols]
        return TemporalData(
            src=data.src[mask],
            dst=data.dst_user[mask],   # dst = post_owner (user node)
            t=data.t[mask].float(),
            msg=msg,
        )

    train = _filt(saved["train"])
    val   = _filt(saved["val"])
    test  = _filt(saved["test"])

    # neg_pool: ALL users in the closed-world fof set
    # (not just seen dst — this is the correct closed-world negative pool)
    neg_pool = torch.tensor(sorted(fof_idx), dtype=torch.long)

    return train, val, test, neg_pool
