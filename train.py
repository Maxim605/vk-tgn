"""
Stage 4: Train TGN for link prediction (u2p or u2u).

Run:
  python learner/train.py              # u2p, no BERT
  python learner/train.py --bert       # u2p, with BERT
  python learner/train.py --u2u        # u2u, no BERT
  python learner/train.py --no-comments --no-degree  # ablations
  python learner/train.py --shuffle-time             # shuffle sanity check
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage, LastAggregator, LastNeighborLoader,
)
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from learner.config import (
    PROC_DIR, CKPT_DIR, RES_DIR, FOF_SEED,
    MEM_DIM, TIME_DIM, EMB_DIM, BATCH_SIZE, N_EPOCHS, LR, PATIENCE, SEED,
    ARANGO_URL, ARANGO_DB, ARANGO_USER, ARANGO_PASS,
)
from learner.models.tgn import GraphAttentionEmbedding, LinkPredictor
from learner.tasks.u2p import build_u2p_splits
from learner.tasks.u2u import build_u2u_splits

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

CKPT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# msg layout in temporal_data.pt (no-bert = 10 dims, bert = 778 dims):
#   no-bert: [temporal(7) | node_feats(3)]
#   bert:    [BERT(768) | temporal(7) | node_feats(3)]
TEMPORAL_BASE  = [0, 1, 2, 3, 4]   # type_w, type_c, dt_user, dt_post, post_age
TEMPORAL_COMM  = [5]                # post_comments (index 5 in temporal block)
NODE_FEAT_COLS = [6, 7, 8]         # degree_norm, out_degree_norm, in_degree_norm


def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fetch_fof_idx() -> set:
    """Query ArangoDB for fof user indices from id_map."""
    import pandas as pd
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
    return fof_idx


def select_msg_cols(use_bert: bool, no_comments: bool, no_degree: bool) -> list:
    """Return list of msg column indices to keep."""
    bert_offset = 768 if use_bert else 0
    temporal_cols = TEMPORAL_BASE + ([] if no_comments else TEMPORAL_COMM)
    cols = [bert_offset + c for c in temporal_cols]
    if use_bert:
        cols = list(range(768)) + cols
    if not no_degree:
        # node feat cols come after temporal block
        n_temporal = len(TEMPORAL_BASE) + (0 if no_comments else 1)
        node_start = bert_offset + n_temporal
        cols += [node_start, node_start + 1, node_start + 2]
    return cols


def run_batch(batch, memory, gnn, link_pred, neighbor_loader,
              hist_t, hist_msg, num_nodes, neg_pool, device):
    batch = batch.to(device)
    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
    neg_dst = neg_pool[torch.randint(0, len(neg_pool), (len(src),), device=device)]

    n_id, edge_index, e_id = neighbor_loader(
        torch.cat([src, dst, neg_dst]).unique()
    )
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    assoc[n_id] = torch.arange(len(n_id), device=device)

    z, last_update = memory(n_id)
    if edge_index.numel() > 0:
        z = gnn(z, last_update, edge_index, hist_t[e_id], hist_msg[e_id])

    pos_out = link_pred(z[assoc[src]], z[assoc[dst]])
    neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])
    return pos_out, neg_out, src, dst, t, msg


def warmup_memory(data, memory, neighbor_loader, device):
    memory.eval()
    with torch.no_grad():
        for batch in TemporalDataLoader(data, batch_size=BATCH_SIZE):
            batch = batch.to(device)
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)


@torch.no_grad()
def evaluate(loader, memory, gnn, link_pred, neighbor_loader,
             hist_t, hist_msg, num_nodes, neg_pool, device):
    memory.eval(); gnn.eval(); link_pred.eval()
    all_y_true, all_y_pred = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        pos_out, neg_out, src, dst, t, msg = run_batch(
            batch, memory, gnn, link_pred, neighbor_loader,
            hist_t, hist_msg, num_nodes, neg_pool, device,
        )
        y_pred = torch.cat([pos_out, neg_out]).sigmoid().cpu().numpy()
        y_true = np.concatenate([np.ones(len(pos_out)), np.zeros(len(neg_out))])
        all_y_true.append(y_true); all_y_pred.append(y_pred)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    return {
        "ap":  float(average_precision_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_pred)),
    }


def train_epoch(loader, memory, gnn, link_pred, neighbor_loader,
                optimizer, criterion, hist_t, hist_msg,
                num_nodes, neg_pool, device):
    memory.train(); gnn.train(); link_pred.train()
    memory.reset_state(); neighbor_loader.reset_state()
    total_loss = 0.0; n_batches = 0
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        pos_out, neg_out, src, dst, t, msg = run_batch(
            batch, memory, gnn, link_pred, neighbor_loader,
            hist_t, hist_msg, num_nodes, neg_pool, device,
        )
        loss = (criterion(pos_out, torch.ones_like(pos_out)) +
                criterion(neg_out, torch.zeros_like(neg_out)))
        loss.backward()
        optimizer.step()
        memory.detach()
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        total_loss += float(loss); n_batches += 1
    return total_loss / max(n_batches, 1)


def run_training(train_data, val_data, test_data,
                 hist_t, hist_msg, num_nodes, neg_pool,
                 msg_dim, device, run_name: str) -> dict:

    neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=device)
    memory = TGNMemory(
        num_nodes, msg_dim, MEM_DIM, TIME_DIM,
        message_module=IdentityMessage(msg_dim, MEM_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
    ).to(device)
    gnn = GraphAttentionEmbedding(
        in_channels=MEM_DIM, out_channels=EMB_DIM,
        msg_dim=msg_dim, time_enc=memory.time_enc,
    ).to(device)
    link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

    seen = set()
    params = []
    for p in list(memory.parameters()) + list(gnn.parameters()) + list(link_pred.parameters()):
        if id(p) not in seen:
            seen.add(id(p)); params.append(p)

    optimizer = torch.optim.Adam(params, lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader   = TemporalDataLoader(val_data,   batch_size=BATCH_SIZE)
    test_loader  = TemporalDataLoader(test_data,  batch_size=BATCH_SIZE)

    best_val_ap = 0.0; patience_cnt = 0; history = []
    ckpt_path = CKPT_DIR / f"{run_name}_best.pt"

    print(f"\n{'='*60}\nTraining [{run_name}]: {N_EPOCHS} epochs\n{'='*60}\n")

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        loss = train_epoch(
            train_loader, memory, gnn, link_pred, neighbor_loader,
            optimizer, criterion, hist_t, hist_msg, num_nodes, neg_pool, device,
        )
        memory.reset_state(); neighbor_loader.reset_state()
        warmup_memory(train_data, memory, neighbor_loader, device)
        val_m = evaluate(val_loader, memory, gnn, link_pred, neighbor_loader,
                         hist_t, hist_msg, num_nodes, neg_pool, device)
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | "
              f"val_ap={val_m['ap']:.4f} | val_auc={val_m['auc']:.4f} | "
              f"{time.time()-t0:.1f}s")
        history.append({"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val_m.items()}})

        if val_m["ap"] > best_val_ap:
            best_val_ap = val_m["ap"]; patience_cnt = 0
            torch.save({"memory": memory.state_dict(), "gnn": gnn.state_dict(),
                        "link_pred": link_pred.state_dict(),
                        "epoch": epoch, "val_ap": best_val_ap}, ckpt_path)
            print(f"  ✅ val_ap={best_val_ap:.4f} → {ckpt_path.name}")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}"); break

    # Final test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    memory.load_state_dict(ckpt["memory"])
    gnn.load_state_dict(ckpt["gnn"])
    link_pred.load_state_dict(ckpt["link_pred"])
    memory.reset_state(); neighbor_loader.reset_state()
    warmup_memory(train_data, memory, neighbor_loader, device)
    warmup_memory(val_data,   memory, neighbor_loader, device)
    test_m = evaluate(test_loader, memory, gnn, link_pred, neighbor_loader,
                      hist_t, hist_msg, num_nodes, neg_pool, device)
    print(f"Test [{run_name}]: ap={test_m['ap']:.4f}  auc={test_m['auc']:.4f}")

    results = {"run": run_name, "msg_dim": msg_dim, "history": history,
               "test": test_m, "best_val_ap": best_val_ap}
    out = RES_DIR / f"tgn_{run_name}_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"→ {out}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert",         action="store_true")
    parser.add_argument("--u2u",          action="store_true")
    parser.add_argument("--no-comments",  action="store_true")
    parser.add_argument("--no-degree",    action="store_true")
    parser.add_argument("--shuffle-time", action="store_true")
    parser.add_argument("--cold-entity",  action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print("Loading temporal_data.pt...")
    saved = torch.load(PROC_DIR / "temporal_data.pt", map_location="cpu",
                       weights_only=False)

    print(f"Fetching fof({FOF_SEED})...")
    fof_idx = fetch_fof_idx()
    print(f"  fof users in graph: {len(fof_idx):,}")

    # Select msg columns based on ablation flags
    msg_cols = select_msg_cols(args.bert, args.no_comments, args.no_degree)

    # Build task splits
    if args.u2u:
        train_data, val_data, test_data, neg_pool = build_u2u_splits(
            saved, fof_idx, msg_cols
        )
    else:
        train_data, val_data, test_data, neg_pool = build_u2p_splits(
            saved, fof_idx, msg_cols
        )

    # Remap to contiguous local IDs
    all_nodes = torch.cat([train_data.src, train_data.dst,
                           val_data.src, val_data.dst,
                           test_data.src, test_data.dst]).unique()
    num_nodes_global = int(saved["full"].num_nodes)
    local_map = torch.full((num_nodes_global,), -1, dtype=torch.long)
    local_map[all_nodes] = torch.arange(len(all_nodes))
    num_nodes = len(all_nodes)

    def remap(data: TemporalData) -> TemporalData:
        return TemporalData(
            src=local_map[data.src], dst=local_map[data.dst],
            t=data.t, msg=data.msg,
        )

    train_data = remap(train_data)
    val_data   = remap(val_data)
    test_data  = remap(test_data)
    neg_pool   = local_map[neg_pool[neg_pool < num_nodes_global]]
    neg_pool   = neg_pool[neg_pool >= 0].to(device)

    msg_dim  = train_data.msg.shape[1]
    hist_t   = torch.cat([train_data.t, val_data.t, test_data.t]).to(device)
    hist_msg = torch.cat([train_data.msg, val_data.msg, test_data.msg]).to(device)

    # Shuffle sanity: permute entire rows (src, dst, msg, t together)
    if args.shuffle_time:
        perm = torch.randperm(train_data.num_events,
                              generator=torch.Generator().manual_seed(SEED))
        train_data = TemporalData(
            src=train_data.src[perm],
            dst=train_data.dst[perm],
            t=train_data.t[perm],
            msg=train_data.msg[perm],
        )
        print("  ⚠️  shuffle_time: entire rows permuted (src, dst, msg, t together)")

    # Cold entity: test only on dst not seen in train
    if args.cold_entity:
        train_dsts = set(train_data.dst.numpy().tolist())
        mask = torch.tensor(
            [int(d) not in train_dsts for d in test_data.dst.numpy()],
            dtype=torch.bool,
        )
        test_data = TemporalData(
            src=test_data.src[mask], dst=test_data.dst[mask],
            t=test_data.t[mask], msg=test_data.msg[mask],
        )
        print(f"  cold_entity: test narrowed to {test_data.num_events:,} events")

    print(f"  num_nodes={num_nodes:,}  msg_dim={msg_dim}  neg_pool={len(neg_pool):,}")
    print(f"  train={train_data.num_events:,}  val={val_data.num_events:,}  "
          f"test={test_data.num_events:,}")

    # Build run name
    parts = ["bert" if args.bert else "no-bert"]
    if args.u2u:          parts.append("u2u")
    if args.no_comments:  parts.append("no-comm")
    if args.no_degree:    parts.append("no-deg")
    if args.shuffle_time: parts.append("shuffle")
    if args.cold_entity:  parts.append("cold")
    run_name = "-".join(parts)

    run_training(
        train_data, val_data, test_data,
        hist_t, hist_msg, num_nodes, neg_pool,
        msg_dim, device, run_name,
    )


if __name__ == "__main__":
    main()
