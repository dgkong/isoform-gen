import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline_model import BaselinePointerHead
from dataset import GeneIterableFixedB, split_filters_by_chroms
from model import PointerHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

PARQUET_DIR = "data/table"

CKPT_DIR = "results"
BASELINE_CKPT = os.path.join(CKPT_DIR, "final_baseline.pt")
HISTORY_CKPT = os.path.join(CKPT_DIR, "final_his.pt")

TX_BATCH_SIZE = 8
PAD_BP = 5000

# bucket edges (bp) for suffix length
SUFFIX_BINS = [0, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 24576, 32768, 65536, 10**9]
# bucket edges (num splice sites) for context length
CTX_BINS = [0, 2, 5, 9, 17, 10**9]

OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

parquets = sorted(Path(PARQUET_DIR).glob("*.parquet"))
if len(parquets) == 0:
    raise FileNotFoundError(f"No parquet files found in {PARQUET_DIR}")
ONE_PARQUET = parquets[0]
print("Using:", ONE_PARQUET)


def bucketize_np(x, edges):
    x = np.asarray(x)
    idx = np.searchsorted(edges, x, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def bin_labels(edges):
    return [f"[{edges[i]},{edges[i+1]})" for i in range(len(edges) - 1)]


@torch.no_grad()
def eval_overall_weighted_and_bins_unweighted(model, loader, suffix_bins, ctx_bins, device):
    model.eval()

    n_suf = len(suffix_bins) - 1
    n_ctx = len(ctx_bins) - 1
    n_roles = 4

    overall_wsum = 0.0
    overall_w = 0.0

    suf_sum = np.zeros(n_suf, dtype=np.float64)
    suf_cnt = np.zeros(n_suf, dtype=np.int64)

    ctx_sum = np.zeros(n_ctx, dtype=np.float64)
    ctx_cnt = np.zeros(n_ctx, dtype=np.int64)

    role_sum = np.zeros(n_roles, dtype=np.float64)
    role_cnt = np.zeros(n_roles, dtype=np.int64)

    for batch_i, batch in enumerate(loader, start=1):
        if batch_i % 10 == 0:
            print(f"processed {batch_i} batches...")

        batch = {k: v.to(device) for k, v in batch.items()}

        logits, _, _ = model(
            x_ctx=batch["X_context"],
            x_suf=batch["X_suffix"],
            pos_ctx=batch["context_pos"],
            pos_suf=batch["suffix_pos"],
            role_ctx=batch["context_roles"],
            role_tgt=batch["target_role"],
            mask_ctx=batch["context_mask"],
            mask_suf=batch["suffix_mask"],
            targets=None,
            rel_abundance=None,
        )

        targets = batch["target_idx"]
        weights = batch["rel_abundance"].float()
        roles   = batch["target_role"].long().cpu().numpy()

        per_ex = F.cross_entropy(logits, targets, reduction="none")

        overall_wsum += float((per_ex * weights).sum().item())
        overall_w += float(weights.sum().item())

        suf_len = batch["suffix_mask"].sum(dim=1).long().cpu().numpy()
        ctx_len = batch["context_mask"].sum(dim=1).long().cpu().numpy()
        loss_np = per_ex.detach().cpu().numpy()

        suf_bin = bucketize_np(suf_len, suffix_bins)
        ctx_bin = bucketize_np(ctx_len, ctx_bins)

        for bs, bc, r, l in zip(suf_bin, ctx_bin, roles, loss_np):
            suf_sum[bs] += float(l)
            suf_cnt[bs] += 1
            ctx_sum[bc] += float(l)
            ctx_cnt[bc] += 1

            if 0 <= r < n_roles:
                role_sum[r] += float(l); role_cnt[r] += 1

    overall_weighted_loss = overall_wsum / max(overall_w, 1e-12)

    return {
        "overall_weighted_loss": overall_weighted_loss,
        "suffix_labels": bin_labels(suffix_bins),
        "suffix_count": suf_cnt,
        "suffix_unweighted_mean_loss": suf_sum / np.maximum(suf_cnt, 1),
        "ctx_labels": bin_labels(ctx_bins),
        "ctx_count": ctx_cnt,
        "ctx_unweighted_mean_loss": ctx_sum / np.maximum(ctx_cnt, 1),
        "role_labels": ["TSS", "D", "A", "PAS"],
        "role_count": role_cnt,
        "role_unweighted_mean_loss": role_sum / np.maximum(role_cnt, 1),
    }


def save_single_model_bins(run_name, stats, out_dir=OUT_DIR):
    df_suf = pd.DataFrame(
        {
            "suffix_bin": stats["suffix_labels"],
            "n": np.asarray(stats["suffix_count"], dtype=int),
            "unweighted_mean_loss": np.asarray(stats["suffix_unweighted_mean_loss"], dtype=float),
        }
    )
    df_suf.to_csv(os.path.join(out_dir, f"{run_name}_suffix_bins.csv"), index=False)

    df_ctx = pd.DataFrame(
        {
            "context_bin": stats["ctx_labels"],
            "n": np.asarray(stats["ctx_count"], dtype=int),
            "unweighted_mean_loss": np.asarray(stats["ctx_unweighted_mean_loss"], dtype=float),
        }
    )
    df_ctx.to_csv(os.path.join(out_dir, f"{run_name}_context_bins.csv"), index=False)

    df_role = pd.DataFrame(
        {
            "target_role": stats["role_labels"],
            "n": np.asarray(stats["role_count"], dtype=int),
            "unweighted_mean_loss": np.asarray(stats["role_unweighted_mean_loss"], dtype=float),
        }
    )
    df_role.to_csv(os.path.join(out_dir, f"{run_name}_role_losses.csv"), index=False)

    summary = pd.DataFrame(
        [
            {
                "run": run_name,
                "overall_weighted_loss": float(stats["overall_weighted_loss"]),
            }
        ]
    )
    summary.to_csv(os.path.join(out_dir, f"{run_name}_summary.csv"), index=False)

    return df_suf, df_ctx, summary

train_filter, val_filter, test_filter = split_filters_by_chroms()

test_dataset = GeneIterableFixedB(
    parquet_dir=PARQUET_DIR,
    split_filter=test_filter,
    tx_batch_size=TX_BATCH_SIZE,
    pad_bp=PAD_BP,
    shuffle=False,
)
test_dataset.parquet_paths = [ONE_PARQUET]
test_loader = DataLoader(test_dataset, batch_size=None)

print("\nLoading baseline:", BASELINE_CKPT)
baseline_model = BaselinePointerHead(
    d_emb=256,
    d_model=512,
    num_roles=4,
    role_dim=64,
    dropout=0.1,
).to(device)
baseline_state = torch.load(BASELINE_CKPT, map_location=device)
baseline_model.load_state_dict(baseline_state["model"])

print("Evaluating baseline...")
baseline_stats = eval_overall_weighted_and_bins_unweighted(
    baseline_model, test_loader, SUFFIX_BINS, CTX_BINS, device
)
save_single_model_bins("baseline", baseline_stats, OUT_DIR)
print("baseline overall weighted loss:", baseline_stats["overall_weighted_loss"])

del baseline_model
if device.type == "cuda":
    torch.cuda.empty_cache()

print("\nLoading history:", HISTORY_CKPT)
history_model = PointerHead(
    d_emb=256,
    d_model=512,
    num_roles=4,
    role_dim=64,
    dropout=0.1,
).to(device)
history_state = torch.load(HISTORY_CKPT, map_location=device)
history_model.load_state_dict(history_state["model"])

print("Evaluating history...")
history_stats = eval_overall_weighted_and_bins_unweighted(
    history_model, test_loader, SUFFIX_BINS, CTX_BINS, device
)
save_single_model_bins("history", history_stats, OUT_DIR)
print("history overall weighted loss:", history_stats["overall_weighted_loss"])

print("\nDone. Wrote CSVs to:", OUT_DIR)
