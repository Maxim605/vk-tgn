"""
Central configuration for the learner pipeline.
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
FOF_SEED = "577933229"

# ── Temporal split fractions ──────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.85

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
WINDOW_SEC = 7 * 24 * 3600
DECAY_HALF = 3 * 24 * 3600

# ── Multi-task head weights ───────────────────────────────────────────────────
HEAD_WEIGHTS = {
    "recipient": 1.0,
    "time":      0.1,
    "action":    0.5,
    "text":      0.1,
}
TIME_BINS     = 32
TEXT_PROJ_DIM = 256

# ── Benchmark difficulty modes ────────────────────────────────────────────────
# easy:      1 random negative, standard chronological split
# medium:    20 negatives (random), standard split
# hard:      100 negatives (random + structural + temporal), standard split
# inductive: hard negatives + inductive split (new users/pairs in test)
BENCHMARK_MODES = {
    "easy": {
        "n_neg":           1,
        "neg_types":       ["random"],
        "split":           "chronological",
        "inductive_frac":  0.0,
        "cold_edge_frac":  0.0,
    },
    "medium": {
        "n_neg":           20,
        "neg_types":       ["random"],
        "split":           "chronological",
        "inductive_frac":  0.0,
        "cold_edge_frac":  0.0,
    },
    "hard": {
        "n_neg":           20,                              # было 100, снижено для скорости
        "neg_types":       ["random", "structural"],        # temporal убран (медленный)
        "split":           "chronological",
        "inductive_frac":  0.0,
        "cold_edge_frac":  0.2,
    },
    "hard100": {
        "n_neg":           100,
        "neg_types":       ["random", "structural", "temporal"],
        "split":           "chronological",
        "inductive_frac":  0.0,
        "cold_edge_frac":  0.2,
    },
    "inductive": {
        "n_neg":           100,
        "neg_types":       ["random", "structural", "temporal"],
        "split":           "inductive",
        "inductive_frac":  0.2,   # 20% test users are new (unseen in train)
        "cold_edge_frac":  0.3,
    },
}

# Ablation head sets
ABLATION_HEADS = {
    "structure":              "recipient",
    "structure+time":         "recipient+time",
    "structure+time+text":    "recipient+time+text",
    "structure+time+text+action": "recipient+time+action+text",
}
