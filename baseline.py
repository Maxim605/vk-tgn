"""
Temporal baselines для user→post и user→user задач.

user→post baselines:
  target_pop_global, target_pop_window_7d, post_recency_decay,
  user_activity_rate, logistic_regression

user→user baselines:
  user_pop, repeated_pair, jaccard_friendship, adamic_adar,
  user_activity_rate, logistic_regression_u2u

Тесты: standard, cold_entity, shuffle_sanity
Негативы: только среди dst существовавших в train.

Запуск:
  python learner/baseline.py           # user→post
  python learner/baseline.py --u2u     # user→user
"""

import argparse
import json
import numpy as np
import torch
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

PROC_DIR   = Path(__file__).parent / "data" / "processed"
RES_DIR    = Path(__file__).parent / "results"
RES_DIR.mkdir(exist_ok=True)

FOF_SEED   = "577933229"
BATCH_SIZE = 5000
SEED       = 42
WINDOW_SEC = 7 * 24 * 3600
DECAY_HALF = 3 * 24 * 3600

# ── Загрузка ──────────────────────────────────────────────────────────────────

def load_splits(use_u2u: bool):
    from arango import ArangoClient

    saved = torch.load(PROC_DIR / "temporal_data.pt", map_location="cpu",
                       weights_only=False)

    arango = ArangoClient(hosts="http://localhost:8529")
    adb = arango.db("_system", username="root", password="test")
    cur = adb.aql.execute("""
        LET u = FIRST(FOR v IN users FILTER v._key == @seed RETURN v)
        LET direct = (FOR f IN 1..1 OUTBOUND u friendships RETURN f._key)
        LET fof = (FOR f IN 1..1 OUTBOUND u friendships
                     FOR ff IN 1..1 OUTBOUND f friendships
                       FILTER ff._key != u._key AND ff._key NOT IN direct
                       RETURN DISTINCT TO_NUMBER(ff._key))
        RETURN fof
    """, bind_vars={"seed": FOF_SEED})
    fof_ids = set(cur.next())

    id_map = pd.read_parquet(PROC_DIR / "id_map.parquet")
    fof_idx = set(id_map[id_map["str_id"].astype(str).isin(
        {str(x) for x in fof_ids})]["idx"].tolist())

    # Friendship graph для u2u baselines
    friendship_src, friendship_dst = [], []
    if use_u2u:
        fr_cur = adb.aql.execute("""
            FOR e IN friendships
              FILTER TO_NUMBER(SPLIT(e._from,'/')[1]) IN @ids
                 AND TO_NUMBER(SPLIT(e._to,'/')[1]) IN @ids
              RETURN {src: TO_NUMBER(SPLIT(e._from,'/')[1]),
                      dst: TO_NUMBER(SPLIT(e._to,'/')[1])}
        """, bind_vars={"ids": list(fof_ids)})
        for row in fr_cur:
            friendship_src.append(row["src"])
            friendship_dst.append(row["dst"])

    def filt(data, use_u2u=False):
        m = torch.tensor([int(s) in fof_idx for s in data.src.numpy()],
                         dtype=torch.bool)
        src = data.src[m]
        t   = data.t[m].float()
        msg = data.msg[m, :1]
        if use_u2u and hasattr(data, 'dst_user'):
            dst_u = data.dst_user[m]
            valid = dst_u >= 0
            return TemporalData(src=src[valid], dst=dst_u[valid],
                                t=t[valid], msg=msg[valid])
        return TemporalData(src=src, dst=data.dst[m], t=t, msg=msg)

    train = filt(saved["train"], use_u2u)
    val   = filt(saved["val"],   use_u2u)
    test  = filt(saved["test"],  use_u2u)
    full  = filt(saved["full"],  use_u2u)

    return train, val, test, full, friendship_src, friendship_dst


# ── Статистики ────────────────────────────────────────────────────────────────

class Stats:
    def __init__(self, train: TemporalData,
                 fr_src: list, fr_dst: list, use_u2u: bool):
        src = train.src.numpy().astype(np.int64)
        dst = train.dst.numpy().astype(np.int64)
        t   = train.t.numpy().astype(np.float64)
        self.use_u2u = use_u2u
        self.t_min   = float(t.min())
        self.t_max   = float(t.max())
        self.t_range = max(self.t_max - self.t_min, 1.0)
        self._src = src; self._dst = dst; self._t = t

        self.dst_count = Counter(dst.tolist())
        self.src_count = Counter(src.tolist())
        self.max_dst   = max(self.dst_count.values(), default=1)
        self.max_src   = max(self.src_count.values(), default=1)

        self.last_t: dict = {}
        for d, ti in zip(dst, t):
            if d not in self.last_t or ti > self.last_t[d]:
                self.last_t[d] = float(ti)

        # User activity rate
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

        # Repeated pair
        self.pair_count = Counter(zip(src.tolist(), dst.tolist()))

        # Friendship adjacency для u2u
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
                scores[i] = float(np.exp(-np.log(2) * max(t_ref - last, 0) / half_life))
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


# ── Оценка ────────────────────────────────────────────────────────────────────

def evaluate(test: TemporalData, score_fn, neg_pool: np.ndarray, rng, name: str):
    all_y_true, all_y_pred = [], []
    for batch in TemporalDataLoader(test, batch_size=BATCH_SIZE):
        src = batch.src.numpy().astype(np.int64)
        dst = batch.dst.numpy().astype(np.int64)
        t   = batch.t.numpy().astype(np.float64)
        neg = neg_pool[rng.integers(0, len(neg_pool), size=len(src))]
        pos_s = score_fn(src, dst, t)
        neg_s = score_fn(src, neg, t)
        all_y_true.append(np.concatenate([np.ones(len(src)), np.zeros(len(src))]))
        all_y_pred.append(np.concatenate([pos_s, neg_s]))
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    ap  = float(average_precision_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_pred))
    print(f"  {name:38s}: ap={ap:.4f}  auc={auc:.4f}")
    return {"name": name, "ap": ap, "auc": auc}


def run_baselines(train, test, neg_pool, rng, stats: Stats, prefix=""):
    results = []
    results.append(evaluate(test, lambda s,d,t: rng.random(len(s)).astype(np.float32),
                             neg_pool, rng, prefix+"random"))

    if stats.use_u2u:
        results.append(evaluate(test,
            lambda s,d,t: np.array([stats.dst_count.get(int(x),0)/stats.max_dst for x in d], dtype=np.float32),
            neg_pool, rng, prefix+"user_popularity"))
        results.append(evaluate(test,
            lambda s,d,t: np.array([stats.pair_count.get((int(si),int(di)),0) for si,di in zip(s,d)], dtype=np.float32),
            neg_pool, rng, prefix+"repeated_pair"))
        results.append(evaluate(test, lambda s,d,t: stats.jaccard(s,d),
                                 neg_pool, rng, prefix+"jaccard_friendship"))
        results.append(evaluate(test, lambda s,d,t: stats.adamic_adar(s,d),
                                 neg_pool, rng, prefix+"adamic_adar"))
        results.append(evaluate(test,
            lambda s,d,t: np.array([stats.user_rate.get(int(x),0)/stats.max_rate for x in s], dtype=np.float32),
            neg_pool, rng, prefix+"user_activity_rate"))
        feat_fn = stats.features_u2u
        lr_name = prefix+"logistic_regression_u2u"
    else:
        results.append(evaluate(test,
            lambda s,d,t: np.array([stats.dst_count.get(int(x),0)/stats.max_dst for x in d], dtype=np.float32),
            neg_pool, rng, prefix+"target_pop_global"))
        results.append(evaluate(test, lambda s,d,t: stats.windowed_pop(d,t),
                                 neg_pool, rng, prefix+"target_pop_window_7d"))
        results.append(evaluate(test,
            lambda s,d,t: stats.recency(d, float(t.mean()) if len(t) else stats.t_max),
            neg_pool, rng, prefix+"post_recency_decay"))
        results.append(evaluate(test,
            lambda s,d,t: np.array([stats.user_rate.get(int(x),0)/stats.max_rate for x in s], dtype=np.float32),
            neg_pool, rng, prefix+"user_activity_rate"))
        feat_fn = stats.features_u2p
        lr_name = prefix+"logistic_regression"

    # Logistic Regression
    print(f"  Обучение LogReg [{prefix}]...")
    rng2 = np.random.default_rng(SEED+1)
    tr_src = train.src.numpy().astype(np.int64)
    tr_dst = train.dst.numpy().astype(np.int64)
    tr_t   = train.t.numpy().astype(np.float64)
    neg_d  = neg_pool[rng2.integers(0, len(neg_pool), size=len(tr_src))]
    X = np.vstack([feat_fn(tr_src, tr_dst, tr_t), feat_fn(tr_src, neg_d, tr_t)])
    y = np.concatenate([np.ones(len(tr_src)), np.zeros(len(tr_src))])
    sc = StandardScaler(); X_s = sc.fit_transform(X)
    lr = LogisticRegression(max_iter=500, C=1.0); lr.fit(X_s, y)
    results.append(evaluate(test,
        lambda s,d,t: lr.predict_proba(sc.transform(feat_fn(s,d,t)))[:,1].astype(np.float32),
        neg_pool, rng, lr_name))
    return results


# ── Cold entity split ─────────────────────────────────────────────────────────

def cold_entity_split(full: TemporalData, frac=0.80):
    n = full.num_events; split = int(n * frac)
    train = TemporalData(src=full.src[:split], dst=full.dst[:split],
                         t=full.t[:split], msg=full.msg[:split])
    seen = set(full.dst[:split].numpy().tolist())
    mask = torch.tensor([int(d) not in seen for d in full.dst[split:].numpy()], dtype=torch.bool)
    test = TemporalData(src=full.src[split:][mask], dst=full.dst[split:][mask],
                        t=full.t[split:][mask], msg=full.msg[split:][mask])
    print(f"  cold_entity: train={train.num_events:,}  test={test.num_events:,}")
    return train, test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--u2u", action="store_true", help="user→user baselines")
    args = parser.parse_args()

    print(f"Загрузка данных (u2u={args.u2u})...")
    train, val, test, full, fr_src, fr_dst = load_splits(args.u2u)
    print(f"  train={train.num_events:,}  val={val.num_events:,}  test={test.num_events:,}")

    rng = np.random.default_rng(SEED)
    neg_pool = test.dst.unique().numpy()  # только dst из test (существовали в train или test)
    # Строже: только dst из train
    neg_pool_train = train.dst.unique().numpy()
    print(f"  neg_pool (train dst): {len(neg_pool_train):,}")

    stats = Stats(train, fr_src, fr_dst, args.u2u)
    all_results = {}

    print("\n=== A. Standard test ===")
    all_results["standard"] = run_baselines(train, test, neg_pool_train, rng, stats, "std_")

    print("\n=== B. Cold entity test ===")
    cold_train, cold_test = cold_entity_split(full)
    if cold_test.num_events > 0:
        cold_neg = cold_train.dst.unique().numpy()
        cold_stats = Stats(cold_train, fr_src, fr_dst, args.u2u)
        all_results["cold_entity"] = run_baselines(
            cold_train, cold_test, cold_neg, rng, cold_stats, "cold_")

    print("\n=== C. Shuffle sanity ===")
    perm = np.random.default_rng(SEED).permutation(train.num_events)
    train_shuf = TemporalData(src=train.src, dst=train.dst,
                              t=train.t[perm], msg=train.msg)
    shuf_stats = Stats(train_shuf, fr_src, fr_dst, args.u2u)
    all_results["shuffle"] = run_baselines(
        train_shuf, test, neg_pool_train, rng, shuf_stats, "shuf_")

    suffix = "_u2u" if args.u2u else ""
    out = RES_DIR / f"baseline{suffix}_metrics.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n→ {out}")

    # Итоговая таблица
    print("\n" + "="*70)
    tgn_files = sorted(RES_DIR.glob("tgn_*_metrics.json"))
    if tgn_files:
        print("TGN:")
        for f in tgn_files:
            r = json.loads(f.read_text())
            print(f"  {f.stem:40s}: ap={r['test']['ap']:.4f}  auc={r['test']['auc']:.4f}")
    print("\nBaselines (standard):")
    for r in all_results.get("standard", []):
        print(f"  {r['name']:40s}: ap={r['ap']:.4f}  auc={r['auc']:.4f}")

    print("\nShuffle sanity:")
    for bname in ["repeated_pair", "user_popularity", "target_pop_window_7d",
                  "post_recency_decay", "logistic_regression"]:
        std  = next((r for r in all_results.get("standard",[]) if bname in r["name"]), None)
        shuf = next((r for r in all_results.get("shuffle",[])  if bname in r["name"]), None)
        if std and shuf:
            d = std["ap"] - shuf["ap"]
            print(f"  {'✅' if d>0.005 else '⚠️ '} {bname:30s}: std={std['ap']:.4f}  shuf={shuf['ap']:.4f}  Δ={d:+.4f}")


if __name__ == "__main__":
    main()
