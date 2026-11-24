"""Shared training utilities for sequence models trained on data/processed/windows.npz."""
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from ml.config import MODELS_DIR, SEQ_LEN, WINDOWS_PATH

SEED = 42
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 60
PATIENCE = 8
GRAD_CLIP = 1.0
HUBER_DELTA = 1.0


def load_windows(path=WINDOWS_PATH) -> dict[str, torch.Tensor | list[str]]:
    """Return X/y/m tensors per split plus feature/target names."""
    data = np.load(path)
    out: dict = {"feature_names": data["feature_names"].tolist(), "target_names": data["target_names"].tolist()}
    for split in ["train", "val", "test"]:
        for key in ["X", "y", "m"]:
            out[f"{key}_{split}"] = torch.from_numpy(data[f"{key}_{split}"])
    return out


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    """Average Huber loss over REAL targets only; masked entries add 0 loss and 0 gradient."""
    per_elem = F.huber_loss(pred, target, reduction="none", delta=delta)
    return (per_elem * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> float:
    model.eval()
    return float(masked_huber_loss(model(X), y, m))


def train_model(model: nn.Module, name: str, model_type: str, hparams: dict, data: dict) -> dict:
    """Train with shuffled mini-batches + early stopping on masked val loss; save best checkpoint."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Shuffle only the TRAIN windows (each window stays internally chronological).
    train_loader = DataLoader(
        TensorDataset(data["X_train"], data["y_train"], data["m_train"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training {name} ({n_params:,} params) on {len(data['X_train']):,} windows | "
          f"val={len(data['X_val']):,} | X shape={tuple(data['X_train'].shape[1:])}")

    best_val, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running, n_batches = 0.0, 0
        for xb, yb, mb in train_loader:
            optimizer.zero_grad()
            loss = masked_huber_loss(model(xb), yb, mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running += loss.item()
            n_batches += 1

        train_loss = running / n_batches
        val_loss = evaluate_loss(model, data["X_val"], data["y_val"], data["m_val"])
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5)})

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"  Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}{'  *best' if improved else ''}")
        if epoch - best_epoch >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    ckpt = {
        "state_dict": best_state,
        "model_type": model_type,
        "hparams": hparams,
        "input_size": len(data["feature_names"]),
        "output_size": len(data["target_names"]),
        "seq_len": SEQ_LEN,
        "feature_cols": data["feature_names"],
        "target_cols": data["target_names"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }
    torch.save(ckpt, MODELS_DIR / f"{name}_model.pt")
    with open(MODELS_DIR / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved {name}_model.pt (best epoch {best_epoch}, masked val Huber {best_val:.4f}) + {name}_history.json")
    return ckpt
