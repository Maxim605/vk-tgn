"""
Минимальная реализация TemporalData без зависимости от torch_geometric.
Совместима с интерфейсом PyG TemporalData.
"""
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TemporalData:
    """
    Поток событий динамического графа.
    src[i], dst[i], t[i], msg[i] — i-е событие.
    """
    src: torch.Tensor        # [E] long
    dst: torch.Tensor        # [E] long
    t:   torch.Tensor        # [E] float64 (unix seconds)
    msg: torch.Tensor        # [E, d_msg] float32
    event_type: Optional[torch.Tensor] = None  # [E] long (0=write, 1=comment)

    @property
    def num_events(self) -> int:
        return self.src.shape[0]

    @property
    def num_nodes(self) -> int:
        return int(max(self.src.max(), self.dst.max()).item()) + 1

    def __getitem__(self, idx):
        """Срез по событиям."""
        return TemporalData(
            src=self.src[idx],
            dst=self.dst[idx],
            t=self.t[idx],
            msg=self.msg[idx],
            event_type=self.event_type[idx] if self.event_type is not None else None,
        )

    def __len__(self) -> int:
        return self.num_events

    def __repr__(self) -> str:
        return (
            f"TemporalData("
            f"num_events={self.num_events}, "
            f"num_nodes={self.num_nodes}, "
            f"msg_dim={self.msg.shape[1]})"
        )

    def to(self, device):
        return TemporalData(
            src=self.src.to(device),
            dst=self.dst.to(device),
            t=self.t.to(device),
            msg=self.msg.to(device),
            event_type=self.event_type.to(device) if self.event_type is not None else None,
        )

    def pin_memory(self):
        return TemporalData(
            src=self.src.pin_memory(),
            dst=self.dst.pin_memory(),
            t=self.t.pin_memory(),
            msg=self.msg.pin_memory(),
            event_type=self.event_type.pin_memory() if self.event_type is not None else None,
        )


class TemporalDataLoader:
    """
    Итератор по батчам событий в хронологическом порядке.
    Каждый батч — последовательный срез событий.
    """
    def __init__(self, data: TemporalData, batch_size: int = 200,
                 neg_sampling_ratio: float = 1.0):
        self.data = data
        self.batch_size = batch_size
        self.neg_sampling