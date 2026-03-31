"""
Hard negative sampler для temporal link prediction benchmark.

Типы негативов:
  random     — случайные узлы из neg_pool (векторизовано, быстро)
  structural — узлы с похожим degree, но без события с src (векторизовано)
  temporal   — пары, которые взаимодействуют в другое время (аппроксимация)

Все методы работают батчами без Python-цикла по событиям.
"""

import numpy as np
import torch
from collections import defaultdict
from torch_geometric.data import TemporalData


class NegativeSampler:
    def __init__(self, train_data: TemporalData, neg_pool: torch.Tensor,
                 neg_types: list, n_neg: int, seed: int = 42):
        self.neg_pool  = neg_pool.numpy().astype(np.int64)
        self.neg_types = neg_types
        self.n_neg     = n_neg
        self.rng       = np.random.default_rng(seed)

        # Распределяем n_neg между типами
        n_types = len(neg_types)
        base    = n_neg // n_types
        rem     = n_neg % n_types
        self._counts = {t: base + (1 if i < rem else 0)
                        for i, t in enumerate(neg_types)}

        if "structural" in neg_types or "temporal" in neg_types:
            self._build_index(train_data)

    def _build_index(self, data: TemporalData):
        src = data.src.numpy().astype(np.int64)
        dst = data.dst.numpy().astype(np.int64)
        t   = data.t.numpy().astype(np.float32)

        # degree по dst
        max_node = int(max(dst.max(), self.neg_pool.max())) + 1
        self._dst_degree = np.bincount(dst, minlength=max_node).astype(np.float32)

        # src → seen dst (для structural: исключаем виденные пары)
        self._src_seen: dict = defaultdict(set)
        for s, d in zip(src, dst):
            self._src_seen[int(s)].add(int(d))

        # temporal: для каждого dst — последнее время события
        self._dst_last_t = np.zeros(max_node, dtype=np.float32)
        for d, ti in zip(dst, t):
            if ti > self._dst_last_t[d]:
                self._dst_last_t[d] = ti

        # Предвычисляем degree для neg_pool
        pool_valid = self.neg_pool[self.neg_pool < max_node]
        self._pool_degree = self._dst_degree[pool_valid]
        self._pool_last_t = self._dst_last_t[pool_valid]
        self._pool_valid  = pool_valid

    def _sample_random(self, B: int, n: int) -> np.ndarray:
        """[B, n] — случайные индексы из neg_pool."""
        idx = self.rng.integers(0, len(self.neg_pool), size=(B, n))
        return self.neg_pool[idx]

    def _sample_structural(self, src_arr: np.ndarray, n: int) -> np.ndarray:
        """
        Полностью векторизованный structural sampler.
        Для каждого src выбираем n узлов из neg_pool с похожим degree.
        """
        B = len(src_arr)

        # Для каждого src: среднее degree виденных dst
        target_degs = np.zeros(B, dtype=np.float32)
        for i, s in enumerate(src_arr):
            seen = self._src_seen.get(int(s), set())
            if seen:
                seen_arr = np.array(list(seen), dtype=np.int64)
                seen_arr = seen_arr[seen_arr < len(self._dst_degree)]
                if len(seen_arr) > 0:
                    target_degs[i] = self._dst_degree[seen_arr].mean()

        # Сортируем neg_pool по degree
        sort_idx      = np.argsort(self._pool_degree)
        sorted_pool   = self._pool_valid[sort_idx]
        sorted_degree = self._pool_degree[sort_idx]
        pool_size     = len(sorted_pool)

        # Позиции в отсортированном массиве для каждого src [B]
        positions = np.searchsorted(sorted_degree, target_degs)

        # Окно: берём n*3 кандидатов вокруг позиции, потом сэмплируем n
        half = min(n * 3, pool_size // 4)
        lo = np.clip(positions - half, 0, pool_size - 2 * half)
        hi = lo + 2 * half

        # [B, 2*half] — индексы в sorted_pool
        window_idx = (lo[:, None] + self.rng.integers(0, 2 * half, size=(B, n))) % pool_size
        return sorted_pool[window_idx]  # [B, n]

    def _sample_temporal(self, t_arr: np.ndarray, n: int) -> np.ndarray:
        """
        Векторизованный temporal sampler.
        Выбираем dst, у которых |last_t - t| > W (взаимодействуют в другое время).
        """
        W = 3600 * 24  # 1 день
        B = len(t_arr)

        # [B, pool_size] — time diff для каждой пары (event, pool_node)
        # Делаем это батчами чтобы не взрывать память
        CHUNK = 500
        result = np.empty((B, n), dtype=np.int64)

        for i in range(0, B, CHUNK):
            t_chunk = t_arr[i:i+CHUNK, None]           # [chunk, 1]
            diffs   = np.abs(self._pool_last_t[None, :] - t_chunk)  # [chunk, pool]
            far     = diffs > W                         # [chunk, pool] bool

            for j in range(len(t_chunk)):
                cands = self._pool_valid[far[j]]
                if len(cands) < n:
                    cands = self._pool_valid
                idx = self.rng.integers(0, len(cands), size=n)
                result[i + j] = cands[idx]

        return result

    def sample(self, src: torch.Tensor, dst: torch.Tensor,
               t: torch.Tensor) -> torch.Tensor:
        """
        Возвращает [B, n_neg] long tensor негативных dst.
        """
        src_np = src.cpu().numpy().astype(np.int64)
        t_np   = t.cpu().numpy().astype(np.float32)
        B      = len(src_np)

        parts = []
        for neg_type in self.neg_types:
            n = self._counts[neg_type]
            if neg_type == "random":
                negs = self._sample_random(B, n)
            elif neg_type == "structural":
                negs = self._sample_structural(src_np, n)
            elif neg_type == "temporal":
                negs = self._sample_temporal(t_np, n)
            else:
                negs = self._sample_random(B, n)
            parts.append(negs)  # [B, n_i]

        return torch.tensor(np.concatenate(parts, axis=1), dtype=torch.long)  # [B, n_neg]
