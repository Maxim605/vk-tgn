"""
Metrics for multi-task temporal event prediction benchmark.

recipient: MRR, Hits@1/5/10/20, AP, AUC  (ranking по N негативам)
time:      MAE (seconds + hours), bin accuracy
action:    macro-F1, balanced accuracy
text:      cosine similarity, Recall@K (retrieval)
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from typing import Optional


def recipient_metrics_ranked(pos_scores: np.ndarray, neg_scores: np.ndarray,
                              k_list: tuple = (1, 5, 10, 20)) -> dict:
    """
    Ranking metrics для случая N негативов на каждый позитив.

    pos_scores: [N]        — score для позитивного события
    neg_scores: [N, n_neg] — scores для n_neg негативов на каждое событие
    """
    assert neg_scores.ndim == 2, "neg_scores должен быть [N, n_neg]"
    N, n_neg = neg_scores.shape

    # Ранг позитива среди [pos, neg1, neg2, ..., neg_n_neg]
    # rank = 1 + число негативов с score >= pos_score
    ranks = 1 + (neg_scores >= pos_scores[:, None]).sum(axis=1)  # [N]

    mrr    = float(np.mean(1.0 / ranks))
    hits   = {f"hits@{k}": float(np.mean(ranks <= k)) for k in k_list}

    # AP/AUC по бинарной постановке (1 pos + n_neg negs)
    y_true = np.concatenate([np.ones(N), np.zeros(N * n_neg)])
    y_pred = np.concatenate([pos_scores, neg_scores.ravel()])
    ap  = float(average_precision_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_pred))

    return {"mrr": mrr, "ap": ap, "auc": auc, **hits,
            "mean_rank": float(ranks.mean()), "median_rank": float(np.median(ranks))}


def time_metrics(pred_log: np.ndarray, true_delta_t: np.ndarray,
                 n_bins: int = 32) -> dict:
    """
    pred_log:     [N] — predicted log(Δt+1)
    true_delta_t: [N] — true Δt in seconds
    """
    pred_sec = np.expm1(np.clip(pred_log, 0, None))
    mae      = float(np.mean(np.abs(pred_sec - true_delta_t)))

    # Bin accuracy: делим log-шкалу на n_bins равных бинов
    log_true = np.log1p(np.clip(true_delta_t, 0, None))
    log_pred = np.clip(pred_log, 0, None)
    max_log  = max(log_true.max(), 1.0)
    bins     = np.linspace(0, max_log, n_bins + 1)
    true_bin = np.digitize(log_true, bins) - 1
    pred_bin = np.digitize(log_pred, bins) - 1
    bin_acc  = float((true_bin == pred_bin).mean())

    return {
        "mae_sec":   mae,
        "mae_hours": mae / 3600,
        "bin_acc":   bin_acc,
    }


def action_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """
    logits: [N, 2]
    labels: [N] — 0=write, 1=comment
    """
    preds = logits.argmax(axis=1)
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    acc      = float((preds == labels).mean())

    # Balanced accuracy
    classes = np.unique(labels)
    bal_acc = float(np.mean([
        (preds[labels == c] == c).mean() for c in classes
    ]))
    return {"macro_f1": macro_f1, "accuracy": acc, "balanced_acc": bal_acc}


def text_metrics(pred_emb: np.ndarray, true_emb: np.ndarray,
                 k_list: tuple = (1, 5, 10)) -> dict:
    """
    pred_emb: [N, D] — predicted embeddings
    true_emb: [N, D] — true BERT embeddings
    """
    pred_n = pred_emb / (np.linalg.norm(pred_emb, axis=1, keepdims=True) + 1e-8)
    true_n = true_emb / (np.linalg.norm(true_emb, axis=1, keepdims=True) + 1e-8)

    cos_sim = float(np.mean((pred_n * true_n).sum(axis=1)))

    # Retrieval Recall@K: для каждого примера ищем ближайших в true_emb
    N = len(pred_n)
    if N <= 5000:  # только для небольших батчей
        scores = pred_n @ true_n.T  # [N, N]
        recall = {}
        for k in k_list:
            top_k = np.argsort(-scores, axis=1)[:, :k]
            hits  = sum(i in top_k[i] for i in range(N))
            recall[f"text_recall@{k}"] = hits / N
    else:
        recall = {f"text_recall@{k}": float("nan") for k in k_list}

    return {"cosine_sim": cos_sim, **recall}


def aggregate_metrics(
    pos_scores:     Optional[np.ndarray] = None,
    neg_scores:     Optional[np.ndarray] = None,   # [N, n_neg]
    time_pred_log:  Optional[np.ndarray] = None,
    time_true:      Optional[np.ndarray] = None,
    action_logits:  Optional[np.ndarray] = None,
    action_labels:  Optional[np.ndarray] = None,
    text_pred:      Optional[np.ndarray] = None,
    text_true:      Optional[np.ndarray] = None,
    n_bins:         int = 32,
) -> dict:
    result = {}
    if pos_scores is not None and neg_scores is not None:
        if neg_scores.ndim == 1:
            neg_scores = neg_scores[:, None]
        result.update(recipient_metrics_ranked(pos_scores, neg_scores))
    if time_pred_log is not None and time_true is not None:
        result.update(time_metrics(time_pred_log, time_true, n_bins))
    if action_logits is not None and action_labels is not None:
        result.update(action_metrics(action_logits, action_labels))
    if text_pred is not None and text_true is not None:
        result.update(text_metrics(text_pred, text_true))
    return result
