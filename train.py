"""
Temporal event benchmark training.

Запуск:
  python learner/train.py --mode easy
  python learner/train.py --mode hard --u2u --bert
  python learner/train.py --mode inductive --heads recipient+time+action+text --bert
  python learner/train.py --ablation   # запускает все 4 ablation режима
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from learner.config import (
    PROC_DIR, CKPT_DIR, RES_DIR, FOF_SEED,
    MEM_DIM, TIME_DIM, EMB_DIM, BATCH_SIZE, N_EPOCHS, LR, PATIENCE, SEED,
    ARANGO_URL, ARANGO_DB, ARANGO_USER, ARANGO_PASS,
    BERT_DIM, TEXT_PROJ_DIM, HEAD_WEIGHTS, BENCHMARK_MODES, ABLATION_HEADS,
)
from learner.models.tgn import GraphAttentionEmbedding
from learner.models.heads import MultiTaskHeads
from learner.models.metrics import aggregate_metrics
from learner.models.negative_sampler import NegativeSampler
from learner.data.benchmark_splits import build_benchmark_splits
from learner.tasks.u2p import build_u2p_splits
from learner.tasks.u2u import build_u2u_splits

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x

CKPT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# msg layout: no-bert=[temporal(6)], bert=[BERT(768)|temporal(6)]
# temporal: type_write, type_comment, dt_user, dt_post, post_age, post_comments
# degree_norm убран — не каузален
TEMPORAL_COLS = [0, 1, 2, 3, 4, 6]   # пропускаем post_likes=0 (индекс 5)
BERT_OFFSET   = 768


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fetch_fof_idx():
    import pandas as pd
    from arango import ArangoClient
    arango = ArangoClient(hosts=ARANGO_URL)
    adb = arango.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)
    cur = adb.aql.execute("""
        LET u = FIRST(FOR v IN users FILTER v._key == @seed RETURN v)
        LET direct = (FOR f IN 1..1 OUTBOUND u friendships RETURN f._key)
        LET fof = (FOR f IN 1..1 OUTBOUND u friendships
                     FOR ff IN 1..1 OUTBOUND f friendships
                       FILTER ff._key != u._key AND ff._key NOT IN direct
                       RETURN DISTINCT TO_STRING(ff._key))
        RETURN fof
    """, bind_vars={"seed": FOF_SEED})
    fof_str_ids = set(cur.next())
    id_map = pd.read_parquet(PROC_DIR / "id_map.parquet")
    return set(id_map[id_map["str_id"].astype(str).isin(fof_str_ids)]["idx"].tolist())


def parse_heads(heads_str):
    active = set(heads_str.split("+"))
    return {k: (v if k in active else 0.0) for k, v in HEAD_WEIGHTS.items()}


def select_msg_cols(use_bert: bool, actual_msg_dim: int = None):
    """
    Возвращает индексы колонок msg для подачи в TGN.
    Если actual_msg_dim задан — проверяет что все индексы в пределах.
    degree_norm убран — не каузален. post_likes=0 убран — бесполезен.
    """
    if use_bert:
        if actual_msg_dim is not None and actual_msg_dim <= BERT_OFFSET:
            # temporal_data.pt собран без BERT — fallback на no-bert
            print(f"  ⚠️  --bert запрошен, но msg_dim={actual_msg_dim} < {BERT_OFFSET}. "
                  f"Запустите: python learner/data/dataset_builder.py (без --no-bert)")
            print(f"  ⚠️  Fallback на no-bert режим")
            return list(TEMPORAL_COLS)
        bert_cols = list(range(BERT_OFFSET))
        temporal  = [BERT_OFFSET + c for c in TEMPORAL_COLS]
        return bert_cols + temporal
    return list(TEMPORAL_COLS)


def extract_targets(batch, use_bert):
    msg = batch.msg
    targets = {}
    dt_idx = BERT_OFFSET + 2 if use_bert else 2
    if dt_idx < msg.shape[1]:
        targets["delta_t"] = msg[:, dt_idx]
    tw_idx = BERT_OFFSET + 0 if use_bert else 0
    if tw_idx < msg.shape[1]:
        targets["action_label"] = (1 - msg[:, tw_idx]).long()
    if use_bert and msg.shape[1] >= BERT_OFFSET:
        targets["text_emb"] = msg[:, :BERT_OFFSET]
    return targets


def warmup_memory(data, memory, neighbor_loader, device):
    memory.eval()
    with torch.no_grad():
        for batch in TemporalDataLoader(data, batch_size=BATCH_SIZE):
            batch = batch.to(device)
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)


def get_node_embeddings(n_id, memory, gnn, neighbor_loader,
                        hist_t, hist_msg, num_nodes, device):
    """Получает эмбеддинги для набора узлов через memory + GNN."""
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    assoc[n_id] = torch.arange(len(n_id), device=device)
    z, last_update = memory(n_id)
    if edge_index.numel() > 0:
        z = gnn(z, last_update, edge_index, hist_t[e_id], hist_msg[e_id])
    return z, assoc


def process_batch(batch, neg_batch, memory, gnn, heads, neighbor_loader,
                  hist_t, hist_msg, num_nodes, device, use_bert, use_infonce):
    """
    batch:     TemporalData батч
    neg_batch: [B, n_neg] long tensor — негативные dst
    """
    batch = batch.to(device)
    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
    neg_dst = neg_batch.to(device)  # [B, n_neg]

    B, n_neg = neg_dst.shape

    # Все уникальные узлы для lookup
    all_nodes = torch.cat([src, dst, neg_dst.reshape(-1)]).unique()
    z, assoc = get_node_embeddings(all_nodes, memory, gnn, neighbor_loader,
                                   hist_t, hist_msg, num_nodes, device)

    z_src     = z[assoc[src]]       # [B, D]
    z_pos_dst = z[assoc[dst]]       # [B, D]

    # [B, n_neg, D]
    neg_flat  = neg_dst.reshape(-1)
    z_neg_dst = z[assoc[neg_flat]].reshape(B, n_neg, -1)

    targets = extract_targets(batch, use_bert)

    total_loss, loss_dict = heads.compute_loss(
        z_src, z_pos_dst, z_neg_dst,
        delta_t=targets.get("delta_t"),
        action_labels=targets.get("action_label"),
        text_emb=targets.get("text_emb"),
        use_infonce=use_infonce,
    )

    return total_loss, loss_dict, z_src, z_pos_dst, z_neg_dst, targets, src, dst, t, msg


def train_epoch(loader, neg_sampler, memory, gnn, heads, neighbor_loader,
                optimizer, hist_t, hist_msg, num_nodes, device, use_bert, use_infonce):
    memory.train(); gnn.train(); heads.train()
    memory.reset_state(); neighbor_loader.reset_state()

    total_loss = 0.0; loss_sums = {}; n_batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        neg_batch = neg_sampler.sample(batch.src, batch.dst, batch.t)  # [B, n_neg]

        loss, loss_dict, z_src, z_pos, z_neg, targets, src, dst, t, msg = \
            process_batch(batch, neg_batch, memory, gnn, heads, neighbor_loader,
                          hist_t, hist_msg, num_nodes, device, use_bert, use_infonce)

        loss.backward()
        optimizer.step()
        memory.detach()
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)

        total_loss += float(loss)
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, {k: v / n for k, v in loss_sums.items()}


@torch.no_grad()
def evaluate(loader, neg_sampler, memory, gnn, heads, neighbor_loader,
             hist_t, hist_msg, num_nodes, device, use_bert, use_infonce):
    memory.eval(); gnn.eval(); heads.eval()

    all_pos, all_neg = [], []
    all_t_pred, all_t_true = [], []
    all_a_logits, all_a_labels = [], []
    all_txt_pred, all_txt_true = [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        neg_batch = neg_sampler.sample(batch.src, batch.dst, batch.t)

        _, _, z_src, z_pos, z_neg, targets, src, dst, t, msg = \
            process_batch(batch, neg_batch, memory, gnn, heads, neighbor_loader,
                          hist_t, hist_msg, num_nodes, device, use_bert, use_infonce)

        B, n_neg, D = z_neg.shape

        # Recipient scores
        pos_s = heads.recipient(z_src, z_pos).sigmoid().cpu().numpy()
        z_src_exp = z_src.unsqueeze(1).expand(B, n_neg, D)
        neg_s = heads.recipient(
            z_src_exp.reshape(B * n_neg, D),
            z_neg.reshape(B * n_neg, D),
        ).sigmoid().reshape(B, n_neg).cpu().numpy()

        all_pos.append(pos_s)
        all_neg.append(neg_s)

        if heads.time is not None and "delta_t" in targets:
            all_t_pred.append(heads.time(z_src).cpu().numpy())
            all_t_true.append(targets["delta_t"].cpu().numpy())

        if heads.action is not None and "action_label" in targets:
            all_a_logits.append(heads.action(z_src).cpu().numpy())
            all_a_labels.append(targets["action_label"].cpu().numpy())

        if heads.text is not None and "text_emb" in targets:
            all_txt_pred.append(heads.text(z_src).cpu().numpy())
            all_txt_true.append(targets["text_emb"].cpu().numpy())

        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)

    return aggregate_metrics(
        pos_scores=np.concatenate(all_pos) if all_pos else None,
        neg_scores=np.concatenate(all_neg) if all_neg else None,
        time_pred_log=np.concatenate(all_t_pred) if all_t_pred else None,
        time_true=np.concatenate(all_t_true) if all_t_true else None,
        action_logits=np.concatenate(all_a_logits) if all_a_logits else None,
        action_labels=np.concatenate(all_a_labels) if all_a_labels else None,
        text_pred=np.concatenate(all_txt_pred) if all_txt_pred else None,
        text_true=np.concatenate(all_txt_true) if all_txt_true else None,
    )


def run_training(train_data, val_data, test_data, neg_pool,
                 hist_t, hist_msg, num_nodes, msg_dim,
                 head_weights, mode_cfg, device, run_name, use_bert):

    n_neg      = mode_cfg["n_neg"]
    neg_types  = mode_cfg["neg_types"]
    use_infonce = n_neg > 1

    # Негативный сэмплер
    neg_sampler_train = NegativeSampler(
        train_data, neg_pool, neg_types, n_neg, seed=SEED)
    neg_sampler_eval  = NegativeSampler(
        train_data, neg_pool, ["random"], n_neg, seed=SEED + 1)

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
    heads = MultiTaskHeads(
        emb_dim=EMB_DIM, bert_dim=BERT_DIM,
        proj_dim=TEXT_PROJ_DIM, head_weights=head_weights,
    ).to(device)

    seen, params = set(), []
    for p in list(memory.parameters()) + list(gnn.parameters()) + list(heads.parameters()):
        if id(p) not in seen:
            seen.add(id(p)); params.append(p)

    optimizer = torch.optim.Adam(params, lr=LR)
    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader   = TemporalDataLoader(val_data,   batch_size=BATCH_SIZE)
    test_loader  = TemporalDataLoader(test_data,  batch_size=BATCH_SIZE)

    # Shape check
    sample = next(iter(train_loader))
    print(f"\n  Shapes (batch={len(sample.src)}):")
    print(f"    src={tuple(sample.src.shape)} dst={tuple(sample.dst.shape)} "
          f"t={tuple(sample.t.shape)} msg={tuple(sample.msg.shape)}")
    print(f"    neg_pool={tuple(neg_pool.shape)}  n_neg={n_neg}  "
          f"use_infonce={use_infonce}")
    print(f"    hist_t={tuple(hist_t.shape)}  hist_msg={tuple(hist_msg.shape)}")
    print(f"    Active heads: {[k for k,v in head_weights.items() if v > 0]}")

    best_val_mrr = 0.0; patience_cnt = 0; history = []
    ckpt_path = CKPT_DIR / f"{run_name}_best.pt"

    print(f"\n{'='*60}\nTraining [{run_name}]\n{'='*60}\n")

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_losses = train_epoch(
            train_loader, neg_sampler_train, memory, gnn, heads, neighbor_loader,
            optimizer, hist_t, hist_msg, num_nodes, device, use_bert, use_infonce,
        )
        memory.reset_state(); neighbor_loader.reset_state()
        warmup_memory(train_data, memory, neighbor_loader, device)
        val_m = evaluate(val_loader, neg_sampler_eval, memory, gnn, heads,
                         neighbor_loader, hist_t, hist_msg, num_nodes,
                         device, use_bert, use_infonce)

        elapsed = time.time() - t0
        losses_str = "  ".join(f"{k}={v:.4f}" for k, v in train_losses.items())
        print(f"Epoch {epoch:02d} | {losses_str} | total={train_loss:.4f}")
        print(f"         | mrr={val_m.get('mrr',0):.4f}  "
              f"hits@10={val_m.get('hits@10',0):.4f}  "
              f"ap={val_m.get('ap',0):.4f}"
              + (f"  mae_h={val_m.get('mae_hours',0):.2f}" if "mae_hours" in val_m else "")
              + (f"  f1={val_m.get('macro_f1',0):.4f}" if "macro_f1" in val_m else "")
              + (f"  cos={val_m.get('cosine_sim',0):.4f}" if "cosine_sim" in val_m else "")
              + f"  [{elapsed:.1f}s]")

        history.append({"epoch": epoch, "loss": train_loss,
                        "train_losses": train_losses, "val": val_m})

        val_mrr = val_m.get("mrr", 0.0)
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr; patience_cnt = 0
            torch.save({"memory": memory.state_dict(), "gnn": gnn.state_dict(),
                        "heads": heads.state_dict(), "epoch": epoch,
                        "val_mrr": best_val_mrr}, ckpt_path)
            print(f"  ✅ val_mrr={best_val_mrr:.4f} → {ckpt_path.name}")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}"); break

    # Финальный тест
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    memory.load_state_dict(ckpt["memory"])
    gnn.load_state_dict(ckpt["gnn"])
    heads.load_state_dict(ckpt["heads"])
    memory.reset_state(); neighbor_loader.reset_state()
    warmup_memory(train_data, memory, neighbor_loader, device)
    warmup_memory(val_data,   memory, neighbor_loader, device)

    test_m = evaluate(test_loader, neg_sampler_eval, memory, gnn, heads,
                      neighbor_loader, hist_t, hist_msg, num_nodes,
                      device, use_bert, use_infonce)

    print(f"\nTest [{run_name}]:")
    print(f"  mrr={test_m.get('mrr',0):.4f}  hits@1={test_m.get('hits@1',0):.4f}  "
          f"hits@5={test_m.get('hits@5',0):.4f}  hits@10={test_m.get('hits@10',0):.4f}  "
          f"hits@20={test_m.get('hits@20',0):.4f}")
    print(f"  ap={test_m.get('ap',0):.4f}  auc={test_m.get('auc',0):.4f}  "
          f"mean_rank={test_m.get('mean_rank',0):.1f}")
    if "mae_hours" in test_m:
        print(f"  mae={test_m['mae_hours']:.2f}h  bin_acc={test_m.get('bin_acc',0):.4f}  "
              f"f1={test_m.get('macro_f1',0):.4f}  cos={test_m.get('cosine_sim',0):.4f}")

    results = {"run": run_name, "msg_dim": msg_dim, "heads": head_weights,
               "mode": mode_cfg, "history": history,
               "test": test_m, "best_val_mrr": best_val_mrr}
    out = RES_DIR / f"tgn_{run_name}_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"→ {out}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    default="hard",
                        choices=list(BENCHMARK_MODES.keys()))
    parser.add_argument("--bert",    action="store_true")
    parser.add_argument("--u2u",     action="store_true")
    parser.add_argument("--heads",   default="recipient+time+action+text")
    parser.add_argument("--ablation", action="store_true",
                        help="Запустить все 4 ablation режима последовательно")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    saved = torch.load(PROC_DIR / "temporal_data.pt", map_location="cpu",
                       weights_only=False)
    fof_idx = fetch_fof_idx()
    print(f"  fof users: {len(fof_idx):,}")

    actual_msg_dim = int(saved["full"].msg.shape[1])
    msg_cols = select_msg_cols(args.bert, actual_msg_dim)
    use_bert_actual = args.bert and actual_msg_dim > BERT_OFFSET

    if args.u2u:
        train_base, val_base, test_base, neg_pool = build_u2u_splits(
            saved, fof_idx, msg_cols)
    else:
        train_base, val_base, test_base, neg_pool = build_u2p_splits(
            saved, fof_idx, msg_cols)

    # Применяем benchmark split поверх базового
    # Собираем full filtered data для benchmark splitter
    full_filtered = TemporalData(
        src=torch.cat([train_base.src, val_base.src, test_base.src]),
        dst=torch.cat([train_base.dst, val_base.dst, test_base.dst]),
        t=torch.cat([train_base.t, val_base.t, test_base.t]),
        msg=torch.cat([train_base.msg, val_base.msg, test_base.msg]),
    )
    splits = build_benchmark_splits(full_filtered, args.mode)
    train_data = splits["train"]
    val_data   = splits["val"]
    test_data  = splits["test"]

    # Remap to contiguous local IDs
    all_nodes = torch.cat([train_data.src, train_data.dst,
                           val_data.src, val_data.dst,
                           test_data.src, test_data.dst]).unique()
    num_nodes_global = int(saved["full"].num_nodes)
    local_map = torch.full((num_nodes_global,), -1, dtype=torch.long)
    local_map[all_nodes] = torch.arange(len(all_nodes))
    num_nodes = len(all_nodes)

    def remap(d):
        return TemporalData(src=local_map[d.src], dst=local_map[d.dst],
                            t=d.t, msg=d.msg)

    train_data = remap(train_data)
    val_data   = remap(val_data)
    test_data  = remap(test_data)
    neg_pool   = local_map[neg_pool[neg_pool < num_nodes_global]]
    neg_pool   = neg_pool[neg_pool >= 0]

    msg_dim  = train_data.msg.shape[1]
    hist_t   = torch.cat([train_data.t, val_data.t, test_data.t]).to(device)
    hist_msg = torch.cat([train_data.msg, val_data.msg, test_data.msg]).to(device)

    print(f"  num_nodes={num_nodes:,}  msg_dim={msg_dim}  neg_pool={len(neg_pool):,}")
    print(f"  train={train_data.num_events:,}  val={val_data.num_events:,}  "
          f"test={test_data.num_events:,}")

    mode_cfg = BENCHMARK_MODES[args.mode]

    # Ablation: запускаем все 4 конфигурации голов
    head_configs = ABLATION_HEADS if args.ablation else {args.heads: args.heads}
    all_results = {}

    for ablation_name, heads_str in head_configs.items():
        head_weights = parse_heads(heads_str)
        if head_weights.get("text", 0) > 0 and not use_bert_actual:
            print(f"  ⚠️  text head требует BERT в msg, отключаем для {ablation_name}")
            head_weights["text"] = 0.0

        parts = [args.mode, "bert" if args.bert else "no-bert"]
        if args.u2u: parts.append("u2u")
        parts.append(ablation_name.replace("+", "-"))
        run_name = "_".join(parts)

        results = run_training(
            train_data, val_data, test_data, neg_pool,
            hist_t, hist_msg, num_nodes, msg_dim,
            head_weights, mode_cfg, device, run_name, use_bert_actual,
        )
        all_results[ablation_name] = results

    # Итоговая таблица
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("ABLATION SUMMARY")
        print("="*70)
        header = f"{'Ablation':35s}  {'MRR':>6}  {'H@10':>6}  {'AP':>6}"
        if any("mae_hours" in r["test"] for r in all_results.values()):
            header += f"  {'MAE_h':>6}  {'F1':>6}  {'Cos':>6}"
        print(header)
        for name, r in all_results.items():
            t = r["test"]
            row = (f"{name:35s}  {t.get('mrr',0):6.4f}  "
                   f"{t.get('hits@10',0):6.4f}  {t.get('ap',0):6.4f}")
            if "mae_hours" in t:
                row += (f"  {t.get('mae_hours',0):6.2f}  "
                        f"{t.get('macro_f1',0):6.4f}  "
                        f"{t.get('cosine_sim',0):6.4f}")
            print(row)

        summary_path = RES_DIR / f"ablation_{args.mode}_summary.json"
        summary_path.write_text(json.dumps(all_results, indent=2))
        print(f"\n→ {summary_path}")


if __name__ == "__main__":
    main()
