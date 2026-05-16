# ============================================================
# train.py — Reusable Training Loop
# Practical Assignment Week 6: CNN Comparison, BN and Transfer Learning
# Dataset: Blood Cell Image Dataset (4 classes)
# Framework: PyTorch
# ============================================================
#
# Contents:
#   BLOCK 1 — Single epoch training
#   BLOCK 2 — Evaluation over a DataLoader
#   BLOCK 3 — Full training loop
#   BLOCK 4 — Checkpoint saving and loading
#
# Usage:
#   from src.train import train
#
#   history = train(
#       model        = model,
#       loader_train = loader_train,
#       loader_val   = loader_val,
#       epochs       = 20,
#       lr           = 1e-3,
#       name         = "LeNet5"
#   )
# ============================================================

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# CONSTANTS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# BLOCK 1 — SINGLE EPOCH TRAINING
# ============================================================

def train_epoch(model: nn.Module,
                loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> tuple:
    """
    Runs a full training epoch.

    Args:
        model     : nn.Module model to train
        loader    : DataLoader for the training set
        criterion : loss function (e.g. CrossEntropyLoss)
        optimizer : optimizer (e.g. Adam, SGD)
        device    : compute device (cpu or cuda)

    Returns:
        (avg_loss, accuracy) for the epoch
    """

    model.train()          # Enable training mode (BN and Dropout active)

    total_loss    = 0.0    # Accumulated epoch loss
    correct       = 0      # Correct prediction count
    total_samples = 0      # Total samples processed

    for images, labels in loader:

        # Move data to the corresponding device
        images = images.to(device)
        labels = labels.to(device)

        # ----- Forward pass -----
        optimizer.zero_grad()           # Clear gradients from previous step
        predictions = model(images)     # Get model logits
        loss = criterion(predictions,   # Compute loss
                         labels)

        # ----- Backward pass -----
        loss.backward()                 # Compute gradients
        optimizer.step()                # Update parameters

        # ----- Accumulate metrics -----
        total_loss    += loss.item() * images.size(0)   # Batch-weighted loss
        pred_classes   = predictions.argmax(dim=1)       # Class with highest logit
        correct       += (pred_classes == labels).sum().item()
        total_samples += images.size(0)

    # Compute epoch averages
    epoch_loss = total_loss / total_samples
    epoch_acc  = correct / total_samples

    return epoch_loss, epoch_acc


# ============================================================
# BLOCK 2 — EVALUATION OVER A DATALOADER
# ============================================================

def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple:
    """
    Evaluates the model over a DataLoader (validation or test).
    Does not update parameters — only computes metrics.

    Args:
        model     : nn.Module model in eval mode
        loader    : validation or test DataLoader
        criterion : loss function
        device    : compute device

    Returns:
        (avg_loss, accuracy) for the evaluated set
    """

    model.eval()           # Disable BN and Dropout for evaluation

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():  # Skip gradient computation (saves memory and time)
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass only
            predictions = model(images)
            loss = criterion(predictions, labels)

            # Accumulate metrics
            total_loss    += loss.item() * images.size(0)
            pred_classes   = predictions.argmax(dim=1)
            correct       += (pred_classes == labels).sum().item()
            total_samples += images.size(0)

    eval_loss = total_loss / total_samples
    eval_acc  = correct / total_samples

    return eval_loss, eval_acc


# ============================================================
# BLOCK 3 — FULL TRAINING LOOP
# ============================================================

def train(model: nn.Module,
          loader_train: DataLoader,
          loader_val: DataLoader,
          epochs: int            = 20,
          lr: float              = 1e-3,
          optimizer_type: str    = "adam",
          name: str              = "Model",
          device: torch.device   = DEVICE,
          save_best: bool        = True,
          checkpoint_dir: str    = "results/checkpoints") -> dict:
    """
    Full training loop with per-epoch metric logging.

    Args:
        model          : nn.Module model to train
        loader_train   : training DataLoader
        loader_val     : validation DataLoader
        epochs         : number of training epochs
        lr             : initial learning rate
        optimizer_type : 'adam' | 'sgd' — optimizer to use
        name           : model name (for logs and checkpoints)
        device         : compute device
        save_best      : if True, saves the best model by val_acc
        checkpoint_dir : folder to save checkpoints

    Returns:
        history : dict with per-epoch metric lists
            {
                'train_loss'   : [...],
                'train_acc'    : [...],
                'val_loss'     : [...],
                'val_acc'      : [...],
                'epoch_time'   : [...],   # seconds per epoch
                'best_val_acc' : float,
                'best_epoch'   : int
            }

    Example usage:
        history = train(
            model        = LeNet5(),
            loader_train = loader_train,
            loader_val   = loader_val,
            epochs       = 20,
            lr           = 1e-3,
            name         = "LeNet5_noBN"
        )
    """

    # Move model to device
    model = model.to(device)

    # ----- Loss function -----
    # CrossEntropyLoss combines LogSoftmax + NLLLoss (standard for classification)
    criterion = nn.CrossEntropyLoss()

    # ----- Optimizer -----
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,       # Standard SGD momentum
            weight_decay=1e-4   # Mild L2 regularization
        )
    else:
        raise ValueError(f"optimizer_type '{optimizer_type}' is not valid. "
                         f"Options: 'adam' | 'sgd'")

    # ----- Scheduler: reduce lr if val_loss stops improving -----
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",      # Reduce when val_loss stops decreasing
        factor=0.5,      # Multiply lr × 0.5
        patience=3       # Wait 3 epochs without improvement before reducing lr
    )

    # ----- Metric history -----
    history = {
        "train_loss"   : [],
        "train_acc"    : [],
        "val_loss"     : [],
        "val_acc"      : [],
        "epoch_time"   : [],
        "best_val_acc" : 0.0,
        "best_epoch"   : 0
    }

    best_val_acc = 0.0   # Track best checkpoint

    # Create checkpoint directory if it does not exist
    if save_best:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ----- Log header -----
    print(f"\n{'=' * 65}")
    print(f"  Training: {name}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {epochs}  |  LR: {lr}  |  Optimizer: {optimizer_type.upper()}")
    print(f"{'=' * 65}")
    print(f"  {'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>9} | {'Val Acc':>8} | {'Time':>7}")
    print(f"  {'-' * 60}")

    # ===== MAIN LOOP =====
    for epoch in range(1, epochs + 1):

        t_start = time.time()

        # --- Training ---
        train_loss, train_acc = train_epoch(
            model, loader_train, criterion, optimizer, device
        )

        # --- Validation ---
        val_loss, val_acc = evaluate(
            model, loader_val, criterion, device
        )

        # --- Scheduler step ---
        scheduler.step(val_loss)

        # --- Epoch time ---
        t_end   = time.time()
        seconds = t_end - t_start

        # --- Log metrics ---
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(seconds)

        # --- Save best model ---
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc
            history["best_epoch"]   = epoch
            marker = " ✓"

            if save_best:
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"{name}_best.pt"
                )
                save_checkpoint(model, optimizer, epoch,
                                history, ckpt_path)

        # --- Per-epoch log ---
        print(f"  {epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
              f"{val_loss:>9.4f} | {val_acc:>7.2%} | {seconds:>5.1f}s{marker}")

    # ----- Final summary -----
    print(f"\n  {'=' * 60}")
    print(f"  ✓ Training complete — {name}")
    print(f"     Best val_acc  : {history['best_val_acc']:.2%} "
          f"(epoch {history['best_epoch']})")
    print(f"     Avg time/epoch: "
          f"{sum(history['epoch_time']) / epochs:.1f}s")
    print(f"  {'=' * 60}\n")

    return history


# ============================================================
# BLOCK 4 — CHECKPOINT SAVING AND LOADING
# ============================================================

def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    history: dict,
                    path: str) -> None:
    """
    Saves model and optimizer state to a .pt checkpoint file.

    Args:
        model     : trained nn.Module model
        optimizer : optimizer with its current state
        epoch     : current epoch number
        history   : dictionary of recorded metrics
        path      : full path of the .pt file to save
    """
    torch.save({
        "epoch"          : epoch,
        "model_state"    : model.state_dict(),       # Model weights
        "optimizer_state": optimizer.state_dict(),   # Optimizer state
        "history"        : history                   # Accumulated metrics
    }, path)


def load_checkpoint(model: nn.Module,
                    path: str,
                    device: torch.device = DEVICE) -> tuple:
    """
    Loads a previously saved checkpoint.

    Args:
        model  : nn.Module model with the same architecture as the checkpoint
        path   : path to the .pt file to load
        device : device to load the weights onto

    Returns:
        (model, history, epoch) — model with loaded weights

    Example usage:
        model, history, epoch = load_checkpoint(
            LeNet5(), 'results/checkpoints/LeNet5_best.pt'
        )
    """
    assert os.path.exists(path), f"Checkpoint not found: {path}"

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])  # Restore weights
    model = model.to(device)

    print(f"✓ Checkpoint loaded: {path}")
    print(f"   Saved epoch  : {checkpoint['epoch']}")
    print(f"   Best val_acc : {checkpoint['history']['best_val_acc']:.2%}")

    return model, checkpoint["history"], checkpoint["epoch"]


# ============================================================
# QUICK VERIFICATION (run directly: python train.py)
# ============================================================

if __name__ == "__main__":

    import sys
    sys.path.append("..")

    from torch.utils.data import TensorDataset

    print("\nVerifying train.py with synthetic data...\n")

    # ----- Create synthetic data -----
    torch.manual_seed(42)

    N_TRAIN, N_VAL = 128, 32
    C, H, W        = 3, 64, 64
    NUM_CLASSES    = 4

    X_train = torch.randn(N_TRAIN, C, H, W)
    y_train = torch.randint(0, NUM_CLASSES, (N_TRAIN,))
    X_val   = torch.randn(N_VAL, C, H, W)
    y_val   = torch.randint(0, NUM_CLASSES, (N_VAL,))

    loader_train = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=32, shuffle=True)
    loader_val   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=32, shuffle=False)

    # ----- Import test model -----
    try:
        from models import LeNet5
    except ImportError:
        # Fallback: minimal model if models.py is not on the path
        class LeNet5(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(C * H * W, NUM_CLASSES)
                )
            def forward(self, x):
                return self.net(x)

    model = LeNet5()

    # ----- Run test training (3 epochs) -----
    history = train(
        model        = model,
        loader_train = loader_train,
        loader_val   = loader_val,
        epochs       = 3,
        lr           = 1e-3,
        name         = "LeNet5_test",
        save_best    = False   # Skip checkpoint saving during test
    )

    # ----- Verify history structure -----
    expected_keys = ["train_loss", "train_acc", "val_loss",
                     "val_acc", "epoch_time",
                     "best_val_acc", "best_epoch"]

    for key in expected_keys:
        assert key in history, f"Missing key in history: {key}"

    assert len(history["train_loss"]) == 3, "Incorrect history length"

    print("✓ History structure verified")
    print(f"   Keys: {list(history.keys())}")
    print(f"   Recorded epochs : {len(history['train_loss'])}")
    print(f"   Best val_acc    : {history['best_val_acc']:.2%}\n")
    print("✓ train.py verified successfully.\n")
