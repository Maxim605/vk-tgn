"""
Multi-task prediction heads for temporal event prediction.

Heads:
  RecipientHead — предсказывает target node (user или post).
                  Asymmetric scorer: concat(z_src, z_dst, z_src*z_dst, |z_src-z_dst|)
                  Loss: BCEWithLogitsLoss (sampled negatives)
                  Metrics: AUC, AP, MRR, Hits@K

  TimeHead      — предсказывает log(Δt + 1) до следующего события.
                  Input: z_src  →  scalar
                  Loss: MSELoss
                  Metrics: MAE (в секундах)

  ActionHead    — предсказывает тип события: write(0) / comment(1).
                  Input: z_src  →  2-class logits
                  Loss: CrossEntropyLoss
                  Metrics: macro-F1

  TextHead      — предсказывает BERT-эмбеддинг следующего текста.
                  Input: z_src  →  projection → BERT_DIM
                  Loss: CosineEmbeddingLoss
                  Metrics: cosine similarity, Recall@K (retrieval)

Все головы принимают node embeddings [B, EMB_DIM] и возвращают лоссы + предсказания.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RecipientHead(nn.Module):
    """
    Asymmetric link scorer.
    Input:  z_src [B, D], z_dst [B, D]
    Output: scalar logit per pair [B]
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        # [B, 4*D] → [B]
        x = torch.cat([z_src, z_dst, z_src * z_dst, (z_src - z_dst).abs()], dim=-1)
        return self.lin(x).squeeze(-1)

    def loss(self, pos_logits: torch.Tensor, neg_logits: torch.Tensor,
             use_infonce: bool = False) -> torch.Tensor:
        """
        pos_logits: [B]
        neg_logits: [B, n_neg] или [B] (1 neg)
        use_infonce: если True — InfoNCE loss (лучше для ranking с N негативами)
        """
        if neg_logits.dim() == 1:
            neg_logits = neg_logits.unsqueeze(1)

        if use_infonce:
            # InfoNCE: softmax over [pos, neg1, ..., neg_n]
            # logits: [B, 1+n_neg]
            all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
            labels = torch.zeros(len(pos_logits), dtype=torch.long,
                                 device=pos_logits.device)
            return torch.nn.functional.cross_entropy(all_logits, labels)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
            # Усредняем по всем негативам
            neg_loss = criterion(neg_logits.reshape(-1),
                                 torch.zeros(neg_logits.numel(), device=neg_logits.device))
            pos_loss = criterion(pos_logits, torch.ones_like(pos_logits))
            return pos_loss + neg_loss


class TimeHead(nn.Module):
    """
    Predicts log(Δt + 1) — time to next event from src node.
    Input:  z_src [B, D]
    Output: scalar [B]
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
        )

    def forward(self, z_src: torch.Tensor) -> torch.Tensor:
        return self.lin(z_src).squeeze(-1)  # [B]

    def loss(self, pred: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: raw seconds → log(Δt + 1)
        target = torch.log1p(delta_t.clamp(min=0).float())
        return F.mse_loss(pred, target)

    @staticmethod
    def mae_seconds(pred_log: torch.Tensor, delta_t: torch.Tensor) -> float:
        pred_sec = torch.expm1(pred_log.clamp(min=0))
        return float(F.l1_loss(pred_sec, delta_t.float()))


class ActionHead(nn.Module):
    """
    Predicts event type: write(0) / comment(1).
    Input:  z_src [B, D]
    Output: logits [B, 2]
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 2),
        )

    def forward(self, z_src: torch.Tensor) -> torch.Tensor:
        return self.lin(z_src)  # [B, 2]

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels.long())


class TextHead(nn.Module):
    """
    Predicts BERT embedding of the next event's text.
    Input:  z_src [B, D]
    Output: projected embedding [B, bert_dim] (L2-normalized)
    Loss:   CosineEmbeddingLoss against actual BERT embedding
    """
    def __init__(self, emb_dim: int, bert_dim: int, proj_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, bert_dim),
        )

    def forward(self, z_src: torch.Tensor) -> torch.Tensor:
        out = self.proj(z_src)
        return F.normalize(out, dim=-1)  # [B, bert_dim]

    def loss(self, pred: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        # target_emb: actual BERT embeddings [B, bert_dim], L2-normalized
        target = F.normalize(target_emb.float(), dim=-1)
        ones = torch.ones(pred.shape[0], device=pred.device)
        return F.cosine_embedding_loss(pred, target, ones)

    @staticmethod
    def cosine_sim(pred: torch.Tensor, target_emb: torch.Tensor) -> float:
        target = F.normalize(target_emb.float(), dim=-1)
        pred_n = F.normalize(pred, dim=-1)
        return float((pred_n * target).sum(dim=-1).mean())


class MultiTaskHeads(nn.Module):
    """
    Container for all prediction heads.
    Active heads controlled by head_weights dict (weight=0 → disabled).

    head_weights example:
      {"recipient": 1.0, "time": 0.1, "action": 0.5, "text": 0.1}
    """
    def __init__(self, emb_dim: int, bert_dim: int,
                 proj_dim: int, head_weights: dict):
        super().__init__()
        self.weights = head_weights
        self.recipient = RecipientHead(emb_dim)
        self.time      = TimeHead(emb_dim)   if head_weights.get("time",   0) > 0 else None
        self.action    = ActionHead(emb_dim) if head_weights.get("action", 0) > 0 else None
        self.text      = TextHead(emb_dim, bert_dim, proj_dim) \
                         if head_weights.get("text", 0) > 0 else None

    def compute_loss(
        self,
        z_src: torch.Tensor,
        z_pos_dst: torch.Tensor,
        z_neg_dst: torch.Tensor,        # [B, n_neg, D] или [B, D]
        delta_t: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        use_infonce: bool = False,
    ) -> tuple:
        losses = {}

        pos_logits = self.recipient(z_src, z_pos_dst)  # [B]

        if z_neg_dst.dim() == 3:
            # [B, n_neg, D] → compute score for each neg
            B, n_neg, D = z_neg_dst.shape
            z_src_exp = z_src.unsqueeze(1).expand(B, n_neg, D)
            neg_logits = self.recipient(
                z_src_exp.reshape(B * n_neg, D),
                z_neg_dst.reshape(B * n_neg, D),
            ).reshape(B, n_neg)  # [B, n_neg]
        else:
            neg_logits = self.recipient(z_src, z_neg_dst)  # [B]

        losses["recipient"] = self.recipient.loss(pos_logits, neg_logits, use_infonce)

        if self.time is not None and delta_t is not None:
            losses["time"] = self.time.loss(self.time(z_src), delta_t)

        if self.action is not None and action_labels is not None:
            losses["action"] = self.action.loss(self.action(z_src), action_labels)

        if self.text is not None and text_emb is not None:
            losses["text"] = self.text.loss(self.text(z_src), text_emb)

        total = sum(self.weights.get(k, 1.0) * v for k, v in losses.items())
        return total, {k: float(v) for k, v in losses.items()}
