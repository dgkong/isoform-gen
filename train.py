import os
import time

import torch
from torch.utils.data import DataLoader

from baseline_model import BaselinePointerHead
from dataset import GeneIterableFixedB, split_filters_by_chroms
from model import PointerHead

PARQUET_DIR = "data/table"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MAX_STEPS = 100_000
VAL_INTERVAL = 500
VAL_STEPS = 100
PATIENCE = 10
MIN_DELTA = 1e-4

CHECKPOINT_INTERVAL = 250

TX_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 16

LR = 2e-4
WEIGHT_DECAY = 1e-3
CLIP_NORM = 1.0
RESUME_TRAINING = False

LOG_FILE = os.path.join(CHECKPOINT_DIR, "log.txt")
if not RESUME_TRAINING:
    open(LOG_FILE, "w").close()

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# data
train_filter, val_filter, test_filter = split_filters_by_chroms()

train_dataset = GeneIterableFixedB(
    parquet_dir=PARQUET_DIR,
    split_filter=train_filter,
    tx_batch_size=TX_BATCH_SIZE,
    pad_bp=5000,
    shuffle=True,
)

val_dataset = GeneIterableFixedB(
    parquet_dir=PARQUET_DIR,
    split_filter=val_filter,
    tx_batch_size=TX_BATCH_SIZE,
    pad_bp=5000,
    shuffle=False,
)

train_loader = DataLoader(train_dataset, batch_size=None)
val_loader = DataLoader(val_dataset, batch_size=None)

# model
# model = PointerHead(
#     d_emb=256,
#     d_model=512,
#     num_roles=4,
#     role_dim=64,
#     dropout=0.1,
# ).to(device)

# baseline model
model = BaselinePointerHead(
    d_emb=256,
    d_model=512,
    num_roles=4,
    role_dim=64,
    dropout=0.1,
).to(device)

def create_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)

optimizer = create_optimizer(model)

# checkpoints
def save_checkpoint(path, step, model, optimizer, best_val_loss, patience_counter):
    state = {
        "step": step,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "patience": patience_counter
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optim"])
    step = state.get("step", 0)
    best_val_loss = state.get("best_val_loss", float("inf"))
    patience = state.get("patience", 0)
    return step, best_val_loss, patience

# validation
def run_validation():
    model.eval()
    val_iter = iter(val_loader)
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for _ in range(VAL_STEPS):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            _, _, loss = model(
                x_ctx=batch["X_context"],
                x_suf=batch["X_suffix"],
                pos_ctx=batch["context_pos"],
                pos_suf=batch["suffix_pos"],
                role_ctx=batch["context_roles"],
                role_tgt=batch["target_role"],
                mask_ctx=batch["context_mask"],
                mask_suf=batch["suffix_mask"],
                targets=batch["target_idx"],
                rel_abundance=batch["rel_abundance"],
            )

            total_loss += loss.item()
            steps += 1

    model.train()
    return total_loss / max(steps, 1)

# training loop
best_val_loss = float("inf")
patience_counter = 0
global_step = 0

if RESUME_TRAINING:
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if os.path.isfile(latest_path):
        global_step, best_val_loss, patience_counter = load_checkpoint(latest_path, model, optimizer)
        print(f"resumed from step {global_step}, patience {patience_counter}, best_val_loss {best_val_loss:.6f}")

train_iter = iter(train_loader)
print(f"starting training for {MAX_STEPS} optimizer steps")

while global_step < MAX_STEPS:
    if global_step > 0 and global_step % CHECKPOINT_INTERVAL == 0:
        latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        save_checkpoint(latest_path, global_step, model, optimizer, best_val_loss, patience_counter)
        
    if global_step > 0 and global_step % VAL_INTERVAL == 0:
        val_loss = run_validation()
        print(f"step {global_step:6d}/{MAX_STEPS} | val_loss {val_loss:.6f}")

        with open(LOG_FILE, "a") as f:
            f.write(f"{global_step} val {val_loss:.6f}\n")

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
            print("validation improved, saving best checkpoint")
            save_checkpoint(best_path, global_step, model, optimizer, best_val_loss, patience_counter)
        else:
            patience_counter += 1
            print(f"no improvement, patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("early stopping")
            break

    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro_step in range(GRAD_ACCUM_STEPS):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        _, _, loss = model(
            x_ctx=batch["X_context"],
            x_suf=batch["X_suffix"],
            pos_ctx=batch["context_pos"],
            pos_suf=batch["suffix_pos"],
            role_ctx=batch["context_roles"],
            role_tgt=batch["target_role"],
            mask_ctx=batch["context_mask"],
            mask_suf=batch["suffix_mask"],
            targets=batch["target_idx"],
            rel_abundance=batch["rel_abundance"],
        )
        
        loss = loss / GRAD_ACCUM_STEPS


        if not torch.isfinite(loss):
            print(f"[ERROR] Non-finite loss at step {global_step}, micro {micro_step}: {loss.item()}")
            raise SystemExit

        loss.backward()
        loss_accum += loss.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    optimizer.step()

    dt = time.time() - t0
    print(
        f"step {global_step:6d}/{MAX_STEPS} | "
        f"loss {loss_accum:.6f} | "
        f"grad_norm {grad_norm:.4f} | "
        f"dt {dt*1000:.1f}ms"
    )
    with open(LOG_FILE, "a") as f:
        f.write(f"{global_step} train {loss_accum:.6f}\n")

    global_step += 1

final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
print(f"training finished, saving final checkpoint to {final_path}")
save_checkpoint(final_path, global_step, model, optimizer, best_val_loss, patience_counter)
