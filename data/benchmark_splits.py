"""
Benchmark split builder.

Modes:
  chronological — стандартный хронологический split (70/15/15)
  inductive     — часть пользователей появляется только в test

Дополнительно:
  cold_edge_frac — доля test-событий, где пара (src,dst) не встречалась в train

Запуск: python learner/data/benchmark_splits.py --mode hard
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import TemporalData

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from learner.config import PROC_DIR, TRAIN_FRAC, VAL_FRAC, SEED, BENCHMARK_MODES


def chronological_split(data: TemporalData, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Хронологический split по временным квантилям."""
    t = data.t.numpy()
    t_train = float(np.quantile(t, train_frac))
    t_val   = float(np.quantile(t, val_frac))

    train_end = int((t <= t_train).sum())
    val_end   = int((t <= t_val).sum())

    train = data[:train_end]
    val   = data[train_end:val_end]
    test  = data[val_end:]
    return train, val, test


def inductive_split(data: TemporalData, inductive_frac: float,
                    train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED):
    """
    Inductive split: inductive_frac пользователей видны только в test.
    Алгоритм:
      1. Хронологический split.
      2. Из test выбираем inductive_frac уникальных src-пользователей,
         которые НЕ встречались в train.
      3. Из оставшихся test-событий оставляем только те, где src — inductive user.
    """
    train, val, test = chronological_split(data, train_frac, val_frac)

    rng = np.random.default_rng(seed)
    train_users = set(train.src.numpy().tolist())

    # Пользователи, которые появляются в test но не в train
    test_src = test.src.numpy()
    new_users = sorted(set(test_src.tolist()) - train_users)

    if not new_users:
        print("  ⚠️  inductive split: нет новых пользователей в test, используем chronological")
        return train, val, test, set()

    # Выбираем inductive_frac от всех test-пользователей
    n_inductive = max(1, int(len(new_users) * inductive_frac))
    inductive_users = set(rng.choice(new_users, size=n_inductive, replace=False).tolist())

    # Оставляем в test только события с inductive src
    mask = torch.tensor([int(s) in inductive_users for s in test_src], dtype=torch.bool)
    test_inductive = TemporalData(
        src=test.src[mask], dst=test.dst[mask],
        t=test.t[mask], msg=test.msg[mask],
    )
    if hasattr(test, 'dst_user'):
        test_inductive.dst_user = test.dst_user[mask]

    print(f"  inductive split: {len(inductive_users):,} новых пользователей, "
          f"test={test_inductive.num_events:,} событий")
    return train, val, test_inductive, inductive_users


def apply_cold_edge_filter(test: TemporalData, train: TemporalData,
                           cold_edge_frac: float) -> TemporalData:
    """
    Оставляет в test только cold_edge_frac событий, где пара (src,dst)
    не встречалась в train. Если cold_edge_frac=0 — возвращает test без изменений.
    """
    if cold_edge_frac <= 0:
        return test

    train_pairs = set(zip(train.src.numpy().tolist(), train.dst.numpy().tolist()))
    test_src = test.src.numpy()
    test_dst = test.dst.numpy()

    cold_mask = torch.tensor(
        [(int(s), int(d)) not in train_pairs for s, d in zip(test_src, test_dst)],
        dtype=torch.bool,
    )
    n_cold = int(cold_mask.sum())
    n_total = test.num_events

    if n_cold == 0:
        print("  ⚠️  cold_edge: нет холодных пар в test")
        return test

    # Берём cold_edge_frac от всех cold событий
    cold_indices = cold_mask.nonzero(as_tuple=True)[0]
    n_keep = max(1, int(len(cold_indices) * cold_edge_frac / max(cold_edge_frac, 1e-6)))
    n_keep = min(n_keep, len(cold_indices))

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(cold_indices), size=n_keep, replace=False)
    keep_idx = cold_indices[chosen]

    result = TemporalData(
        src=test.src[keep_idx], dst=test.dst[keep_idx],
        t=test.t[keep_idx], msg=test.msg[keep_idx],
    )
    if hasattr(test, 'dst_user'):
        result.dst_user = test.dst_user[keep_idx]

    print(f"  cold_edge: {n_cold:,}/{n_total:,} холодных пар → оставляем {n_keep:,}")
    return result


def build_benchmark_splits(data: TemporalData, mode: str = "hard") -> dict:
    """
    Строит сплиты для заданного benchmark mode.
    Возвращает dict с ключами: train, val, test, meta.
    """
    cfg = BENCHMARK_MODES[mode]
    print(f"\nBenchmark mode: {mode}")
    print(f"  n_neg={cfg['n_neg']}  neg_types={cfg['neg_types']}")
    print(f"  split={cfg['split']}  inductive_frac={cfg['inductive_frac']}")
    print(f"  cold_edge_frac={cfg['cold_edge_frac']}")

    inductive_users = set()

    if cfg["split"] == "inductive" and cfg["inductive_frac"] > 0:
        train, val, test, inductive_users = inductive_split(
            data, cfg["inductive_frac"]
        )
    else:
        train, val, test = chronological_split(data)

    if cfg["cold_edge_frac"] > 0:
        test = apply_cold_edge_filter(test, train, cfg["cold_edge_frac"])

    meta = {
        "mode": mode,
        "n_train": train.num_events,
        "n_val":   val.num_events,
        "n_test":  test.num_events,
        "n_inductive_users": len(inductive_users),
        "config": cfg,
    }
    print(f"  train={train.num_events:,}  val={val.num_events:,}  test={test.num_events:,}")
    return {"train": train, "val": val, "test": test, "meta": meta,
            "inductive_users": inductive_users}
