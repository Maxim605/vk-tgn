"""
Central configuration for the learner pipeline.
All constants, paths, and hyperparameters live here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
LEARNER_DIR = Path(__file__).parent
RAW_DIR     = LEARNER_DIR / "data" / "raw"
PROC_DIR    = LEARNER_DIR / "data" / "processed"
CKPT_DIR    = LEARNER_DIR / "checkpoints"
RES_DIR     = LEARNER_DIR / "results"
MODEL_CACHE = LEARNER_DIR / "data" / "model_cache"

# ── ArangoDB ──────────────────────────────────────────────────────────────────
ARANGO_URL  = "http://localhost:8529"
ARANGO_DB   = "_system"
ARANGO_USER = "root"
ARANGO_PASS = "test"

# ── Graph / task ──────────────────────────────────────────────────────────────
FOF_SEED = "577933229"   # seed user for friends-of-friends subgraph

# ── Temporal split fractions ──────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.85   # 0.70..0.85 = val, 0.85..1.0 = test

# ── Model hyperparameters ─────────────────────────────────────────────────────
MEM_DIM    = 100
TIME_DIM   = 100
EMB_DIM    = 100
BATCH_SIZE = 1000
N_EPOCHS   = 50
LR         = 1e-4
PATIENCE   = 5

# ── BERT ──────────────────────────────────────────────────────────────────────
BERT_MODEL  = "DeepPavlov/rubert-base-cased"
BERT_DIM    = 768
BERT_BATCH  = 256
BERT_MAXLEN = 128

# ── Misc ──────────────────────────────────────────────────────────────────────
SEED       = 42
WINDOW_SEC = 7 * 24 * 3600   # 7-day window for popularity features
DECAY_HALF = 3 * 24 * 3600   # half-life for recency decay
