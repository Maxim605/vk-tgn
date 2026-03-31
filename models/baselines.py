"""
Baseline models for u2p and u2u link prediction.

All baselines are stateless scorers that take (src, dst, t) arrays
and return a float32 score array. No model code in data files.

Baselines:
  u2p: random, target_pop_global, target_pop_window_7d,
       post_recency_decay, user_activity_rate, logistic_regression
  u2u: random, user_popularity, repeated_pair,
       jaccard_friendship, adamic_adar, user_activity_rate,
       logistic_regression_u2u
"""

import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import TemporalData

from learner.config import WINDOW_SEC, DECAY_HALF, SEED


class Stats:
    """Precomputed statistics from training data for baseline scoring."""

    def __init__(self, train: TemporalData,
                 fr_src: list, fr_dst: list, use_u2u: bool):
        src = train.src.numpy().astype(np.int64)
        dst = train.dst.numpy().astype(np.int64)
        t   = train.t.numpy().astype(np.float64)
        self.use_u2u = use_u2u
        self.t_min   = float(t.min())
        self.t_max   = float(t.max())
        self._src = src; self._dst = dst; self._t = t

        self.dst_count = Counter(dst.tolist())
        self.src_count = Counter(src.tolist())
        self.max_dst   = max(self.dst_count.values(), default=1)
        self.max_src   = max(self.src_count.values(), default=1)

        self.last_t: dict = {}
        for d, ti in zip(dst, t):
            if d not in self.last_t or ti > self.last_t[d]:
                self.last_t[d] = float(ti)

        user_span: dict = {}
        user_cnt: dict  = {}
        for s, ti in zip(src, t):
            if s not in user_span:
                user_span[s] = [ti, ti]
            else:
                user_span[s][0] = min(user_span[s][0], ti)
                user_span[s][1] = max(user_span[s][1], ti)
            user_cnt[s] = user_cnt.get(s, 0) + 1
        self.user_rate: dict = {}
        for s, (t0, t1) in user_span.items():
            self.user_rate[s] = user_cnt[s] / max(t1 - t0, 86400.0)
        self.max_rate = max(self.user_rate.values(), default=1.0)

        self.pair_count = Counter(zip(src.tolist(), dst.tolist()))

        self.fr_nbrs: dict = defaultdict(set)
        for s, d in zip(fr_src, fr_dst):
            self.fr_nbrs[s].add(d)
            self.fr_nbrs[d].add(s)

    def windowed_pop(self, dst_arr, t_arr, window=WINDOW_SEC):
        scores = np.zeros(len(dst_arr), dtype=np.float32)
        for i, (d, ti) in enumerate(zip(dst_arr, t_arr)):
            mask = (self._dst == d) & (self._t >= ti - window) & (self._t < ti)
            scores[i] = float(mask.sum())
        return scores

    def recency(self, dst_arr, t_ref, half_life=DECAY_HALF):
        scores = np.zeros(len(dst_arr), dtype=np.float32)
        for i, d in enumerate(dst_arr):
            last = self.last_t.get(int(d))
            if last is not None:
                scores[i] = float(
                    np.exp(-np.log(2) * max(t_ref - last, 0) / half_life)
                )
        return scores

    def jaccard(self, src_arr, dst_arr):
        scores = np.zeros(len(src_arr), dtype=np.float32)
        for i, (s, d) in enumerate(zip(src_arr, dst_arr)):
            ns = self.fr_nbrs.get(int(s), set())
            nd = self.fr_nbrs.get(int(d), set())
            union = len(ns | nd)
            if union > 0:
                scores[i] = len(ns & nd) / union
        return scores

    def adamic_adar(self, src_arr, dst_arr):
        scores = np.zeros(len(src_arr), dtype=np.float32)
        for i, (s, d) in enumerate(zip(src_arr, dst_arr)):
            ns = self.fr_nbrs.get(int(s), set())
            nd = self.fr_nbrs.get(int(d), set())
            for z in ns & nd:
                deg = len(self.fr_nbrs.get(z, set()))
                if deg > 1:
                    scores[i] += 1.0 / np.log(deg)
        return scores

    def features_u2p(self, src_arr, dst_arr, t_arr):
        gp  = np.array([self.dst_count.get(int(d), 0) / self.max_dst for d in dst_arr], dtype=np.float32)
        wp  = self.windowed_pop(dst_arr, t_arr)
        wp  = wp / max(wp.max(), 1.0)
        rec = self.recency(dst_arr, float(t_arr.mean()) if len(t_arr) else self.t_max)
        ur  = np.array([self.user_rate.get(int(s), 0) / self.max_rate for s in src_arr], dtype=np.float32)
        rp  = np.array([self.pair_count.get((int(s), int(d)), 0) for s, d in zip(src_arr, dst_arr)], dtype=np.float32)
        return np.nan_to_num(np.stack([gp, wp, rec, ur, rp], axis=1))

    def features_u2u(self, src_arr, dst_arr, t_arr):
        up  = np.array([self.dst_count.get(int(d), 0) / self.max_dst for d in dst_arr], dtype=np.float32)
        rp  = np.array([self.pair_count.get((int(s), int(d)), 0) for s, d in zip(src_arr, dst_arr)], dtype=np.float32)
        jac = self.jaccard(src_arr, dst_arr)
        aa  = self.adamic_adar(src_arr, dst_arr)
        ur  = np.array([self.user_rate.get(int(s), 0) / self.max_rate for s in src_arr], dtype=np.float32)
        return np.nan_to_num(np.stack([up, rp, jac, aa, ur], axis=1))


def fit_logreg(train: TemporalData, neg_pool: np.ndarray,
               feat_fn, rng) -> tuple:
    """Fit logistic regression on train data. Returns (model, scaler)."""
    rng2 = np.random.default_rng(SEED + 1)
    tr_src = train.src.numpy().astype(np.int64)
    tr_dst = train.dst.numpy().astype(np.int64)
    tr_t   = train.t.numpy().astype(np.float64)
    neg_d  = neg_pool[rng2.integers(0, len(neg_pool), size=len(tr_src))]
    X = np.vstack([feat_fn(tr_src, tr_dst, tr_t), feat_fn(tr_src, neg_d, tr_t)])
    y = np.concatenate([np.ones(len(tr_src)), np.zeros(len(tr_src))])
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X_s, y)
    return lr, sc
