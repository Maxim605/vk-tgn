"""
Stage 2: Build a single canonical event stream from raw parquet files.

Responsibilities:
  - Filter events where src ∈ fof_ids (friends-of-friends subgraph)
  - For u2u: dst = post_owner; filter events where owner also ∈ fof_ids (closed world)
  - Build unified ID space: users [0..N_u), posts [N_u..N_u+N_p)
  - Mark external users (present in events but not in users table)
  - Save canonical stream as parquet + id_map

Outputs (learner/data/processed/):
  canonical_events.parquet  — filtered, ID-mapped event stream
  id_map.parquet            — str_id → int idx mapping

Run: python learner/data/canonicalize.py
"""

import sys
from pathlib import Path

import pandas as pd
from arango import ArangoClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from learner.config import (
    ARANGO_URL, ARANGO_DB, ARANGO_USER, ARANGO_PASS,
    RAW_DIR, PROC_DIR, FOF_SEED,
)

PROC_DIR.mkdir(parents=True, exist_ok=True)


def fetch_fof_ids() -> set:
    """Query ArangoDB for friends-of-friends of FOF_SEED."""
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
    return set(cur.next())


def build_canonical():
    print("=== canonicalize.py ===\n")

    print("Loading raw data...")
    events_df = pd.read_parquet(RAW_DIR / "events.parquet")
    posts_df  = pd.read_parquet(RAW_DIR / "posts.parquet")
    users_df  = pd.read_parquet(RAW_DIR / "users.parquet")
    print(f"  events: {len(events_df):,}  posts: {len(posts_df):,}  users: {len(users_df):,}")

    print(f"\nFetching fof({FOF_SEED}) from ArangoDB...")
    fof_ids = fetch_fof_ids()
    print(f"  fof users: {len(fof_ids):,}")

    # ── Filter: src must be in fof ────────────────────────────────────────────
    events_df = events_df.dropna(subset=["t"]).copy()
    events_df = events_df[events_df["src"].astype(str).isin(fof_ids)].copy()
    events_df = events_df.sort_values("t").reset_index(drop=True)
    print(f"\nAfter fof filter: {len(events_df):,} events")

    # ── Post owner lookup ─────────────────────────────────────────────────────
    post_owner = posts_df.set_index("id")["owner_id"].to_dict()
    events_df["post_owner"] = events_df["dst"].astype(str).map(post_owner)

    # ── ID space ──────────────────────────────────────────────────────────────
    # Users: known (in users table) + external (in events but not in users)
    known_users   = set(users_df["id"].astype(str))
    event_users   = set(events_df["src"].astype(str))
    # Also include post owners so u2u dst IDs are always in the map
    owner_users   = set(str(v) for v in post_owner.values() if v is not None)
    all_users     = sorted(known_users | event_users | owner_users)
    external_users = (event_users | owner_users) - known_users

    all_posts = sorted(set(events_df["dst"].astype(str)))

    user2idx = {uid: i for i, uid in enumerate(all_users)}
    post2idx = {pid: i + len(user2idx) for i, pid in enumerate(all_posts)}

    print(f"\nID space:")
    print(f"  known users:    {len(known_users):,}")
    print(f"  external users: {len(external_users):,}")
    print(f"  posts:          {len(all_posts):,}")
    print(f"  total nodes:    {len(user2idx) + len(post2idx):,}")

    # ── Save id_map ───────────────────────────────────────────────────────────
    id_map = pd.DataFrame(
        [
            {
                "str_id": k,
                "idx": v,
                "node_type": "user",
                "is_external": k in external_users,
            }
            for k, v in user2idx.items()
        ]
        + [
            {"str_id": k, "idx": v, "node_type": "post", "is_external": False}
            for k, v in post2idx.items()
        ]
    )
    id_map.to_parquet(PROC_DIR / "id_map.parquet", index=False)

    # ── Map IDs ───────────────────────────────────────────────────────────────
    events_df["src_idx"]      = events_df["src"].astype(str).map(user2idx)
    events_df["dst_idx"]      = events_df["dst"].astype(str).map(post2idx)
    events_df["dst_user_idx"] = events_df["post_owner"].astype(str).map(user2idx)

    # u2u closed-world flag: owner must also be in fof
    events_df["u2u_valid"] = events_df["post_owner"].astype(str).isin(fof_ids)

    missing = events_df["src_idx"].isna().sum() + events_df["dst_idx"].isna().sum()
    print(f"\nMissing IDs after mapping: {missing}")

    events_df = events_df.dropna(subset=["src_idx", "dst_idx"])
    events_df["src_idx"] = events_df["src_idx"].astype(int)
    events_df["dst_idx"] = events_df["dst_idx"].astype(int)

    # ── Save canonical stream ─────────────────────────────────────────────────
    out_cols = [
        "src", "dst", "src_idx", "dst_idx", "dst_user_idx",
        "t", "type", "text", "comment_id", "owner_id",
        "post_owner", "u2u_valid",
    ]
    out_cols = [c for c in out_cols if c in events_df.columns]
    events_df[out_cols].to_parquet(PROC_DIR / "canonical_events.parquet", index=False)

    print(f"\n✅ Canonical stream saved")
    print(f"   canonical_events: {len(events_df):,} rows")
    print(f"   id_map:           {len(id_map):,} nodes")
    print(f"   → {PROC_DIR}")

    return events_df, id_map, user2idx, post2idx


if __name__ == "__main__":
    build_canonical()
