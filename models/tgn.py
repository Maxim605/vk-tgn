"""
TGN (Temporal Graph Network) model components.

Components:
  GraphAttentionEmbedding — TransformerConv with time encoding
  LinkPredictor           — asymmetric: concat(z_src, z_dst, z_src*z_dst, |z_src-z_dst|)

No data loading here — models receive tensors only.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GraphAttentionEmbedding(nn.Module):
    """
    Single-layer graph attention embedding using TransformerConv.
    Edge features = [time_encoding | msg].
    """

    def __init__(self, in_channels: int, out_channels: int,
                 msg_dim: int, time_enc: nn.Module):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2,
            heads=2, dropout=0.1, edge_dim=edge_dim,
        )

    def forward(self, x, last_update, edge_index, t, msg):
        # rel_t: time since last update for each source node
        rel_t = t - last_update[edge_index[0]]
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(nn.Module):
    """
    Asymmetric link predictor.
    Input: concat(z_src, z_dst, z_src * z_dst, |z_src - z_dst|)
    This captures both symmetric similarity and directed interaction signal.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # Asymmetric: concat(z_src, z_dst, z_src*z_dst, |z_src-z_dst|)
        self.lin = nn.Sequential(
            nn.Linear(in_channels * 4, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [z_src, z_dst, z_src * z_dst, (z_src - z_dst).abs()], dim=-1
        )
        return self.lin(x).squeeze(-1)
