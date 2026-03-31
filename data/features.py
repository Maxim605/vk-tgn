"""
Этап 1.2-1.3: Temporal features, нормализация, ID mapping.

Задача: user → post interaction prediction (user пишет/комментирует пост).
Для user → user link prediction: dst = post_owner (из posts.owner_id).

Запуск: python learner/data/features.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR  = Path(__file__).parent / "raw"
PROC_DIR = Path(__file__).parent / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Загрузка ──────────────────────────────────────────────────────────────────
print("Загрузка данных...")
events_df = pd.read_parquet(RAW_DIR / "events.parquet")
posts_df  = pd.read_parquet(RAW_DIR / "posts.parquet")
users_df  = pd.read_parquet(RAW_DIR / "users.parquet")

print(f"  events: {len(events_df):,}")
print(f"  posts:  {len(posts_df):,}")
print(f"  users:  {len(users_df):,}")

# ── Фильтрация: только события с timestamp ────────────────────────────────────
events_df = events_df.dropna(subset=["t"]).copy()
events_df = events_df.sort_values("t").reset_index(drop=True)
print(f"\nПосле фильтрации: {len(events_df):,} событий")
print(f"  Временной диапазон: {pd.to_datetime(events_df.t.min(), unit='s')} → {pd.to_datetime(events_df.t.max(), unit='s')}")

# ── ID mapping ────────────────────────────────────────────────────────────────
# Единое пространство: users [0..N_u), posts [N_u..N_u+N_p)
print("\nПостроение ID mapping...")

all_src = events_df["src"].unique()
all_dst = events_df["dst"].unique()

# Пользователи: из users_df + внешние (есть в events, нет в users)
known_users = set(users_df["id"].astype(str))
event_users = set(all_src.astype(str))
external_users = event_users - known_users

all_users = sorted(known_users | event_users)
all_posts = sorted(set(all_dst.astype(str)))

user2idx = {uid: i for i, uid in enumerate(all_users)}
post2idx = {pid: i + len(user2idx) for i, pid in enumerate(all_posts)}

print(f"  Пользователей (known):   {len(known_users):,}")
print(f"  Пользователей (external): {len(external_users):,}")
print(f"  Постов:                  {len(all_posts):,}")
print(f"  Всего узлов:             {len(user2idx) + len(post2idx):,}")

# Сохраняем маппинг
id_map = pd.DataFrame([
    {"str_id": k, "idx": v, "node_type": "user",
     "is_external": k in external_users}
    for k, v in user2idx.items()
] + [
    {"str_id": k, "idx": v, "node_type": "post", "is_external": False}
    for k, v in post2idx.items()
])
id_map.to_parquet(PROC_DIR / "id_map.parquet", index=False)

# ── Применяем маппинг к событиям ─────────────────────────────────────────────
events_df["src_idx"] = events_df["src"].astype(str).map(user2idx)
events_df["dst_idx"] = events_df["dst"].astype(str).map(post2idx)

# user→user: dst_user_idx = owner пользователь поста (для user→user задачи)
post_owner = posts_df.set_index("id")["owner_id"].to_dict()
events_df["dst_user"] = events_df["dst"].astype(str).map(post_owner)
events_df["dst_user_idx"] = events_df["dst_user"].astype(str).map(user2idx)

# Проверка
missing = events_df["src_idx"].isna().sum() + events_df["dst_idx"].isna().sum()
print(f"\nПропущенных ID после маппинга: {missing}")
missing_uu = events_df["dst_user_idx"].isna().sum()
print(f"Пропущенных dst_user_idx (нет owner в users): {missing_uu:,} / {len(events_df):,}")

events_df = events_df.dropna(subset=["src_idx", "dst_idx"])
events_df["src_idx"] = events_df["src_idx"].astype(int)
events_df["dst_idx"] = events_df["dst_idx"].astype(int)

# ── Temporal features (строго до t — groupby + shift) ─────────────────────────
print("\nВычисление temporal features...")

# delta_t_user: время с прошлого события пользователя
events_df["delta_t_user"] = (
    events_df.groupby("src_idx")["t"].diff().fillna(0)
)

# delta_t_post: время с прошлого события поста
events_df["delta_t_post"] = (
    events_df.groupby("dst_idx")["t"].diff().fillna(0)
)

# post_age: возраст поста в момент события
post_date = posts_df.set_index("id")["date"].to_dict()
events_df["post_created_at"] = events_df["dst"].astype(str).map(post_date)
events_df["post_age"] = (
    events_df["t"] - pd.to_numeric(events_df["post_created_at"], errors="coerce")
).clip(lower=0).fillna(0)

# type_onehot
events_df["type_write"]   = (events_df["type"] == "write").astype(float)
events_df["type_comment"] = (events_df["type"] == "comment").astype(float)

print(f"  delta_t_user: mean={events_df.delta_t_user.mean():.0f}s  median={events_df.delta_t_user.median():.0f}s")
print(f"  delta_t_post: mean={events_df.delta_t_post.mean():.0f}s  median={events_df.delta_t_post.median():.0f}s")
print(f"  post_age:     mean={events_df.post_age.mean()/86400:.1f}d  median={events_df.post_age.median()/86400:.1f}d")

# ── Нормализация temporal features ───────────────────────────────────────────
# Считаем статистики ТОЛЬКО по train (первые 70%) — leakage prevention
n = len(events_df)
train_end = int(n * 0.70)

train_median_delta_user = events_df.iloc[:train_end]["delta_t_user"].median()
train_median_delta_post = events_df.iloc[:train_end]["delta_t_post"].median()
train_median_post_age   = events_df.iloc[:train_end]["post_age"].median()

# log(1 + x / median) — масштабирует к ~1 для типичных значений
def norm_temporal(x, median):
    return np.log1p(x / (median + 1e-6))

events_df["delta_t_user_norm"] = norm_temporal(events_df["delta_t_user"], train_median_delta_user)
events_df["delta_t_post_norm"] = norm_temporal(events_df["delta_t_post"], train_median_delta_post)
events_df["post_age_norm"]     = norm_temporal(events_df["post_age"],     train_median_post_age)

# ── Статические фичи постов ───────────────────────────────────────────────────
print("\nНормализация фичей постов...")

posts_df["likes_norm"] = 0.0  # likes без timestamp → leakage, обнуляем
posts_df["text_len"]   = posts_df["text"].str.len().fillna(0)
posts_df["text_len_norm"] = np.log1p(posts_df["text_len"])
posts_df = posts_df.drop_duplicates(subset=["id"], keep="first")

# post_comments_norm — строго каузально: для каждого события считаем
# сколько comment-событий на тот же dst было ДО текущего момента t.
# Используем groupby + cumcount на отсортированном потоке.
print("  Вычисление causal post_comments_norm (cumcount до t)...")
comment_mask = events_df["type"] == "comment"
# cumcount по dst среди comment-событий — строго до текущей строки (shift=1)
events_df["post_comments_so_far"] = (
    events_df[comment_mask]
    .groupby("dst_idx")
    .cumcount()  # 0-based: сколько предыдущих comment-событий на этот пост
)
events_df["post_comments_so_far"] = events_df["post_comments_so_far"].fillna(0)

# Нормализуем по train-медиане
train_median_comments = events_df.iloc[:train_end]["post_comments_so_far"].median()
events_df["post_comments_norm"] = norm_temporal(
    events_df["post_comments_so_far"], train_median_comments
)
events_df["post_likes_norm"] = 0.0  # likes без timestamp → 0

# ── Нормализация фичей пользователей ─────────────────────────────────────────
print("Нормализация фичей пользователей...")

users_df["degree_norm"]     = np.log1p(users_df["degree"].fillna(0))
users_df["out_degree_norm"] = np.log1p(users_df["out_degree"].fillna(0))
users_df["in_degree_norm"]  = np.log1p(users_df["in_degree"].fillna(0))
users_df["is_external"] = ~users_df["id"].astype(str).isin(known_users)

# Добавляем внешних пользователей с нулевыми фичами
external_df = pd.DataFrame({
    "id": list(external_users),
    "out_degree": 0, "in_degree": 0, "degree": 0,
    "degree_norm": 0.0, "out_degree_norm": 0.0, "in_degree_norm": 0.0,
    "is_external": True,
})
users_full_df = pd.concat([users_df, external_df], ignore_index=True)
users_full_df["idx"] = users_full_df["id"].astype(str).map(user2idx)

# ── Temporal split по времени (не по индексу) ────────────────────────────────
# Все события с одинаковым t остаются в одном сплите — нет boundary leakage.
t_sorted = np.sort(events_df["t"].values)
t_train  = float(np.quantile(t_sorted, 0.70))
t_val    = float(np.quantile(t_sorted, 0.85))

# Сдвигаем границы вправо до конца группы одинаковых timestamp
train_end = int((events_df["t"] <= t_train).sum())
val_end   = int((events_df["t"] <= t_val).sum())
n         = len(events_df)

split_info = {
    "n_events":  n,
    "train_end": train_end,
    "val_end":   val_end,
    "test_end":  n,
    "train_t_max": float(events_df.iloc[train_end - 1]["t"]),
    "val_t_max":   float(events_df.iloc[val_end - 1]["t"]),
    "test_t_max":  float(events_df.iloc[-1]["t"]),
}
print(f"\nTemporal split:")
print(f"  train: 0..{train_end:,}  t_max={pd.to_datetime(split_info['train_t_max'], unit='s')}")
print(f"  val:   {train_end:,}..{val_end:,}  t_max={pd.to_datetime(split_info['val_t_max'], unit='s')}")
print(f"  test:  {val_end:,}..{n:,}  t_max={pd.to_datetime(split_info['test_t_max'], unit='s')}")

# ── Сохранение ────────────────────────────────────────────────────────────────
# Выбираем только нужные колонки для events
event_cols = [
    "src_idx", "dst_idx", "dst_user_idx", "t", "type", "text",
    "type_write", "type_comment",
    "delta_t_user_norm", "delta_t_post_norm", "post_age_norm",
    "post_likes_norm", "post_comments_norm",
]
events_df[event_cols].to_parquet(PROC_DIR / "events_featured.parquet", index=False)

posts_df.to_parquet(PROC_DIR / "post_features.parquet", index=False)
users_full_df.to_parquet(PROC_DIR / "node_features.parquet", index=False)

with open(PROC_DIR / "split_info.json", "w") as f:
    json.dump(split_info, f, indent=2)

print(f"\n✅ features.py завершён")
print(f"   events_featured: {len(events_df):,} строк, {len(event_cols)} колонок")
print(f"   node_features:   {len(users_full_df):,} узлов")
print(f"   → {PROC_DIR}")
