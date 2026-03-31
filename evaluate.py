"""
Evaluation script: run all baselines and compare with saved TGN results.

Run:
  python learner/evaluate.py           # u2p baselines
  python learner/evaluate.py --u2u     # u2u baselines
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from learner.config import (
    PROC_DIR, RES_DIR, FOF_SEED, BATCH_SIZE, SEED,
    ARANGO_URL, ARANGO_DB, ARANGO_USER, ARANGO_PASS,
)
from learner.models.baselines import Stats, fit_logreg
from learner.tasks.u2p import build_u2p_splits
from learner.tasks.u2u import build_u2u_splits

RES_DIR.mkdir(exist_ok=True)


def fetch_fof_data(use_u2u: bool):
    """Fetch fof_idx and friendship graph from ArangoDB."""
    from arango import ArangoClient

    arango = ArangoClient(hosts=ARANGO_URL)
    adb = arango.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)
    cur = adb.aql.execute(
        """
        LET u = FIRST(FOR v IN users FILTER v._key == @seed RETURN v)
        LET direct = (FOR f IN 1..1 OUTBOUND u friendships RETURN f._key)
        LET fof = (
            FOR f IN 1..1 OUTBOUND u friendships
              FOR ff IN 1..1 OUTBOUND f friendships
                FILTER ff._key != u._key AND ff._key NOT IN direct
                RETURN DISTINCT TO_STRING(ff._key)
        )
        RETURN fof
        """,
        bind_vars={"seed": FOF_SEED},
    )
    fof_str_ids = set(cur.next())

    id_map = pd.read_parquet(PROC_DIR / "id_map.parquet")
    fof_idx = set(
        id_map[id_map["str_id"].astype(str).isin(fof_str_ids)]["idx"].tolist()
    )

    fr_src, fr_dst = [], []
    if use_u2u:
        fr_cur = adb.aql.execute(
            """
            FOR e IN friendships
              FILTER TO_NUMBER(SPLIT(e._from,'/')[1]) IN @ids
                 AND TO_NUMBER(SPLIT(e._to,'/')[1]) IN @ids
              RETURN {src: TO_NUMBER(SPLIT(e._from,'/')[1]),
                      dst: TO_NUMBER(SPLIT(e._to,'/')[1])}
            """,
            bind_vars={"ids": [int(x) for x in fof_str_ids if x.isdigit()]},
        )
        for row in fr_cur:
            fr_src.append(row["src"]); fr_dst.append(row["dst"])

    return fof_idx, fr_src, fr_dst


def evaluate_scorer(test: TemporalData, score_fn, neg_pool: np.ndarray,
                    rng, name: str) -> dict:
    all_y_true, all_y_pred = [], []
    for batch in TemporalDataLoader(test, batch_size=BATCH_SIZE):
        src = batch.src.numpy().astype(np.int64)
        dst = batch.dst.numpy().astype(np.int64)
        t   = batch.t.numpy().astype(np.float64)
        neg = neg_pool[rng.integers(0, len(neg_pool), size=len(src))]
        all_y_true.append(np.concatenate([np.ones(len(src)), np.zeros(len(src))]))
        all_y_pred.append(np.concatenate([score_fn(src, dst, t),
                                           score_fn(src, neg, t)]))
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    ap  = float(average_precision_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_pred))
    print(f"  {name:40s}: ap={ap:.4f}  auc={auc:.4f}")
    return {"name": name, "ap": ap, "auc": auc}


def run_baselines(train, test, neg_pool_np, rng, stats: Stats, prefix=""):
    results = []
    results.append(evaluate_scorer(
        test, lambda s, d, t: rng.random(len(s)).astype(np.float32),
        neg_pool_np, rng, prefix + "random",
    ))

    if stats.use_u2u:
        results.append(evaluate_scorer(test,
            lambda s, d, t: np.array([stats.dst_count.get(int(x), 0) / stats.max_dst for x in d], dtype=np.float32),
            neg_pool_np, rng, prefix + "user_popularity"))
        results.append(evaluate_scorer(test,
            lambda s, d, t: np.array([stats.pair_count.get((int(si), int(di)), 0) for si, di in zip(s, d)], dtype=np.float32),
            neg_pool_np, rng, prefix + "repeated_pair"))
        results.append(evaluate_scorer(test, lambda s, d, t: stats.jaccard(s, d),
                                        neg_pool_np, rng, prefix + "jaccard_friendship"))
        results.append(evaluate_scorer(test, lambda s, d, t: stats.adamic_adar(s, d),
                                        neg_pool_np, rng, prefix + "adamic_adar"))
        results.append(evaluate_scorer(test,
            lambda s, d, t: np.array([stats.user_rate.get(int(x), 0) / stats.max_rate for x in s], dtype=np.float32),
            neg_pool_np, rng, prefix + "user_activity_rate"))
        feat_fn = stats.features_u2u; lr_name = prefix + "logistic_regression_u2u"
    else:
        results.append(evaluate_scorer(test,
            lambda s, d, t: np.array([stats.dst_count.get(int(x), 0) / stats.max_dst for x in d], dtype=np.float32),
            neg_pool_np, rng, prefix + "target_pop_global"))
        results.append(evaluate_scorer(test, lambda s, d, t: stats.windowed_pop(d, t),
                                        neg_pool_np, rng, prefix + "target_pop_window_7d"))
        results.append(evaluate_scorer(test,
            lambda s, d, t: stats.recency(d, float(t.mean()) if len(t) else stats.t_max),
            neg_pool_np, rng, prefix + "post_recency_decay"))
        results.append(evaluate_scorer(test,
            lambda s, d, t: np.array([stats.user_rate.get(int(x), 0) / stats.max_rate for x in s], dtype=np.float32),
            neg_pool_np, rng, prefix + "user_activity_rate"))
        feat_fn = stats.features_u2p; lr_name = prefix + "logistic_regression"

    print(f"  Training LogReg [{prefix}]...")
    lr, sc = fit_logreg(train, neg_pool_np, feat_fn, rng)
    results.append(evaluate_scorer(test,
        lambda s, d, t: lr.predict_proba(sc.transform(feat_fn(s, d, t)))[:, 1].astype(np.float32),
        neg_pool_np, rng, lr_name))
    return results


def cold_entity_split(full: TemporalData, frac: float = 0.80):
    n = full.num_events; split = int(n * frac)
    train = TemporalData(src=full.src[:split], dst=full.dst[:split],
                         t=full.t[:split], msg=full.msg[:split])
    seen = set(full.dst[:split].numpy().tolist())
    mask = torch.tensor([int(d) not in seen for d in full.dst[split:].numpy()],
                        dtype=torch.bool)
    test = TemporalData(src=full.src[split:][mask], dst=full.dst[split:][mask],
                        t=full.t[split:][mask], msg=full.msg[split:][mask])
    print(f"  cold_entity: train={train.num_events:,}  test={test.num_events:,}")
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--u2u", action="store_true")
    args = parser.parse_args()

    print(f"Loading data (u2u={args.u2u})...")
    saved = torch.load(PROC_DIR / "temporal_data.pt", map_location="cpu",
                       weights_only=False)

    fof_idx, fr_src, fr_dst = fetch_fof_data(args.u2u)
    print(f"  fof users: {len(fof_idx):,}")

    if args.u2u:
        train, val, test, neg_pool_t = build_u2u_splits(saved, fof_idx)
    else:
        train, val, test, neg_pool_t = build_u2p_splits(saved, fof_idx)

    # Build full filtered stream for cold split
    if args.u2u:
        full, _, _, _ = build_u2u_splits(saved, fof_idx)
    else:
        full, _, _, _ = build_u2p_splits(saved, fof_idx)

    print(f"  train={train.num_events:,}  val={val.num_events:,}  test={test.num_events:,}")

    rng = np.random.default_rng(SEED)
    neg_pool_np = train.dst.unique().numpy()
    stats = Stats(train, fr_src, fr_dst, args.u2u)
    all_results = {}

    print("\n=== A. Standard test ===")
    all_results["standard"] = run_baselines(train, test, neg_pool_np, rng, stats, "std_")

    print("\n=== B. Cold entity test ===")
    cold_train, cold_test = cold_entity_split(full)
    if cold_test.num_events > 0:
        cold_neg = cold_train.dst.unique().numpy()
        cold_stats = Stats(cold_train, fr_src, fr_dst, args.u2u)
        all_results["cold_entity"] = run_baselines(
            cold_train, cold_test, cold_neg, rng, cold_stats, "cold_"
        )

    print("\n=== C. Shuffle sanity ===")
    perm = np.random.default_rng(SEED).permutation(train.num_events)
    train_shuf = TemporalData(src=train.src, dst=train.dst,
                               t=train.t[perm], msg=train.msg)
    shuf_stats = Stats(train_shuf, fr_src, fr_dst, args.u2u)
    all_results["shuffle"] = run_baselines(
        train_shuf, test, neg_pool_np, rng, shuf_stats, "shuf_"
    )

    suffix = "_u2u" if args.u2u else ""
    out = RES_DIR / f"baseline{suffix}_metrics.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n→ {out}")

    # Summary table
    print("\n" + "=" * 70)
    tgn_files = sorted(RES_DIR.glob("tgn_*_metrics.json"))
    if tgn_files:
        print("TGN results:")
        for f in tgn_files:
            r = json.loads(f.read_text())
            print(f"  {f.stem:42s}: ap={r['test']['ap']:.4f}  auc={r['test']['auc']:.4f}")
    print("\nBaselines (standard):")
    for r in all_results.get("standard", []):
        print(f"  {r['name']:42s}: ap={r['ap']:.4f}  auc={r['auc']:.4f}")
    print("\nShuffle sanity (std vs shuf Δap):")
    for bname in ["repeated_pair", "user_popularity", "target_pop_window_7d",
                  "post_recency_decay", "logistic_regression"]:
        std  = next((r for r in all_results.get("standard", []) if bname in r["name"]), None)
        shuf = next((r for r in all_results.get("shuffle",  []) if bname in r["name"]), None)
        if std and shuf:
            d = std["ap"] - shuf["ap"]
            flag = "✅" if d > 0.005 else "⚠️ "
            print(f"  {flag} {bname:32s}: std={std['ap']:.4f}  shuf={shuf['ap']:.4f}  Δ={d:+.4f}")


if __name__ == "__main__":
    main()
