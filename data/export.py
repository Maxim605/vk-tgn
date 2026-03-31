"""
Stage 1: Export raw data from ArangoDB → parquet files.

Outputs (learner/data/raw/):
  events.parquet  — write/comment event stream with texts
  posts.parquet   — static post features
  users.parquet   — static user features (degree)

Run: python learner/data/export.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
from arango import ArangoClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from learner.config import (
    ARANGO_URL, ARANGO_DB, ARANGO_USER, ARANGO_PASS, RAW_DIR,
)

RAW_DIR.mkdir(parents=True, exist_ok=True)

client = ArangoClient(hosts=ARANGO_URL)
db = client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)


def query_all(aql: str, bind_vars: dict = None, timeout: int = 600) -> list:
    cursor = db.aql.execute(
        aql, bind_vars=bind_vars or {}, batch_size=5000, max_runtime=timeout
    )
    rows = []
    while True:
        rows.extend(cursor.batch())
        if cursor.has_more():
            cursor.fetch()
        else:
            break
    return rows


# ── 1. Interactions ───────────────────────────────────────────────────────────
print("1/5 Exporting interactions...")
t0 = time.time()

interactions_raw = query_all("""
    FOR e IN interactions
      FILTER e.type IN ['write', 'comment']
      FILTER STARTS_WITH(e._from, 'users/')
      RETURN {
        src:        SPLIT(e._from, '/')[1],
        dst:        SPLIT(e._to,   '/')[1],
        t:          DATE_TIMESTAMP(e.created_at) / 1000.0,
        type:       e.type,
        comment_id: TO_STRING(e.comment_id),
        owner_id:   TO_STRING(SPLIT(e._to, '/')[1])
      }
""")

interactions_df = pd.DataFrame(interactions_raw)
interactions_df["t"] = pd.to_numeric(interactions_df["t"], errors="coerce")
# Filter out negative src (groups that leaked through)
interactions_df = interactions_df[
    pd.to_numeric(interactions_df["src"], errors="coerce") > 0
].copy()
print(f"   {len(interactions_df):,} rows  ({time.time()-t0:.1f}s)")


# ── 2. Posts ──────────────────────────────────────────────────────────────────
print("2/5 Exporting posts...")
t0 = time.time()

posts_raw = query_all("""
    FOR p IN posts
      RETURN {
        id:         p._key,
        owner_id:   p.owner_id != null ? TO_STRING(p.owner_id) : TO_STRING(p.ownerId),
        text:       p.text,
        date:       p.date,
        has_attach: p.attachments != null AND LENGTH(p.attachments) > 0
      }
""")

posts_df = pd.DataFrame(posts_raw)
posts_df["text"] = posts_df["text"].fillna("")
posts_df["date"] = pd.to_numeric(posts_df["date"], errors="coerce")
print(f"   {len(posts_df):,} rows  ({time.time()-t0:.1f}s)")


# ── 3. Comments ───────────────────────────────────────────────────────────────
print("3/5 Exporting comments...")
t0 = time.time()

comments_raw = query_all("""
    FOR c IN comment
      RETURN {
        comment_id: TO_STRING(c.comment_id),
        post_id:    TO_STRING(c.post_id),
        owner_id:   TO_STRING(c.owner_id),
        from_id:    TO_STRING(c.from_id),
        text:       c.text
      }
""")

comments_df = pd.DataFrame(comments_raw)
comments_df["text"] = comments_df["text"].fillna("")
# Composite key: owner_id + post_id + comment_id — guarantees uniqueness
comments_df["join_key"] = (
    comments_df["owner_id"].astype(str) + "_" +
    comments_df["post_id"].astype(str)  + "_" +
    comments_df["comment_id"].astype(str)
)
comment_by_key = comments_df.set_index("join_key")["text"].to_dict()
# Fallback: by comment_id alone
comment_by_cid = (
    comments_df.drop_duplicates("comment_id")
    .set_index("comment_id")["text"]
    .to_dict()
)
print(f"   {len(comments_df):,} rows  ({time.time()-t0:.1f}s)")
print(f"   Unique composite keys: {len(comment_by_key):,}")


# ── 4. Likes (static, no timestamp) ──────────────────────────────────────────
print("4/5 Counting likes per post (static)...")
t0 = time.time()

likes_raw = query_all("""
    FOR e IN interactions
      FILTER e.type == 'like'
      COLLECT post_key = SPLIT(e._to, '/')[1] WITH COUNT INTO cnt
      RETURN { id: post_key, likes_count: cnt }
""")

likes_df = (
    pd.DataFrame(likes_raw)
    if likes_raw
    else pd.DataFrame(columns=["id", "likes_count"])
)
print(f"   {len(likes_df):,} posts with likes  ({time.time()-t0:.1f}s)")


# ── 5. Users — only those needed (src + post owners) ─────────────────────────
# Load only users needed for u2u: src users AND post owners.
# This avoids loading all 38M users.
print("5/5 Exporting users (src + post owners only)...")
t0 = time.time()

all_needed_users = set(interactions_df["src"].astype(str)) | set(
    posts_df["owner_id"].astype(str)
)
active_user_ids = list(all_needed_users)
print(f"   Users needed: {len(active_user_ids):,}")

BATCH = 10_000
users_raw = []
for i in range(0, len(active_user_ids), BATCH):
    batch_ids = active_user_ids[i : i + BATCH]
    rows = query_all(
        """
        FOR uid IN @ids
          LET u = FIRST(FOR v IN users FILTER v._key == uid RETURN v)
          FILTER u != null
          LET out_deg = LENGTH(FOR e IN friendships FILTER e._from == u._id RETURN 1)
          LET in_deg  = LENGTH(FOR e IN friendships FILTER e._to   == u._id RETURN 1)
          RETURN { id: uid, out_degree: out_deg, in_degree: in_deg }
        """,
        bind_vars={"ids": batch_ids},
    )
    users_raw.extend(rows)
    print(f"   {min(i+BATCH, len(active_user_ids)):,}/{len(active_user_ids):,}", end="\r")

users_df = pd.DataFrame(users_raw)
for col in ["out_degree", "in_degree"]:
    users_df[col] = pd.to_numeric(users_df[col], errors="coerce").fillna(0).astype(int)
users_df["degree"] = users_df["out_degree"] + users_df["in_degree"]
print(f"   {len(users_df):,} users  ({time.time()-t0:.1f}s)")


# ── Join comment texts ────────────────────────────────────────────────────────
print("\nJoining texts...")

post_text = posts_df.set_index("id")["text"].to_dict()


def get_text(row):
    if row["type"] == "write":
        return post_text.get(str(row["dst"]), "")
    else:
        # FIX: use owner_id from the interaction event (not dst twice)
        key = f"{row['owner_id']}_{row['dst']}_{row['comment_id']}"
        text = comment_by_key.get(key, "")
        if not text:
            text = comment_by_cid.get(str(row["comment_id"]), "")
        return text


interactions_df["text"] = interactions_df.apply(get_text, axis=1)
interactions_df["text"] = interactions_df["text"].fillna("")
interactions_df = interactions_df.sort_values("t").reset_index(drop=True)

n_empty = (interactions_df["text"] == "").sum()
n_comment_empty = (
    interactions_df[interactions_df["type"] == "comment"]["text"] == ""
).sum()
print(f"  Events without text: {n_empty:,} / {len(interactions_df):,}")
print(f"  Comments without text: {n_comment_empty:,} / "
      f"{(interactions_df['type']=='comment').sum():,}")


# ── Join post statics ─────────────────────────────────────────────────────────
posts_df = posts_df.merge(likes_df, on="id", how="left")
posts_df["likes_count"] = posts_df["likes_count"].fillna(0).astype(int)

comments_per_post = (
    interactions_df[interactions_df["type"] == "comment"]
    .groupby("dst")
    .size()
    .reset_index(name="comments_count")
    .rename(columns={"dst": "id"})
)
posts_df = posts_df.merge(comments_per_post, on="id", how="left")
posts_df["comments_count"] = posts_df["comments_count"].fillna(0).astype(int)


# ── Save ──────────────────────────────────────────────────────────────────────
interactions_df.to_parquet(RAW_DIR / "events.parquet", index=False)
posts_df.to_parquet(RAW_DIR / "posts.parquet", index=False)
users_df.to_parquet(RAW_DIR / "users.parquet", index=False)

print(f"\n✅ Export complete")
print(f"   events.parquet:  {len(interactions_df):,} rows")
print(f"   posts.parquet:   {len(posts_df):,} rows")
print(f"   users.parquet:   {len(users_df):,} rows")
print(f"   → {RAW_DIR}")
