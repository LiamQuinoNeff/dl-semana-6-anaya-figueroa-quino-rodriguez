# ============================================================
# utils.py — Auxiliary Functions
# Practical Assignment Week 6: CNN Comparison, BN and Transfer Learning
# Dataset: Blood Cell Image Dataset (4 classes)
# Framework: PyTorch
# ============================================================
#
# Contents:
#   BLOCK 1 — Training curves
#   BLOCK 2 — Multi-model comparison
#   BLOCK 3 — Confusion matrix
#   BLOCK 4 — Batch Normalization analysis
#   BLOCK 5 — Test set predictions and metrics
#
# Usage:
#   from src.utils import (
#       plot_curves, plot_model_comparison,
#       plot_confusion_matrix, plot_bn_convergence,
#       get_predictions, epoch_for_accuracy
#   )
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# CONSTANTS
# ============================================================

CLASSES     = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
FIGURES_DIR = "results/figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# BLOCK 1 — TRAINING CURVES
# ============================================================

def plot_curves(history: dict,
                name: str = "Model",
                save: bool = True,
                figures_dir: str = FIGURES_DIR) -> None:
    """
    Plots loss and accuracy curves (train and validation)
    from the history dict returned by train.train().

    Args:
        history     : dict with keys 'train_loss', 'val_loss',
                      'train_acc', 'val_acc'
        name        : model name (title and filename)
        save        : if True, saves the figure to figures_dir
        figures_dir : destination folder for figures
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training Curves — {name}",
                 fontsize=13, fontweight="bold")

    # ----- Subplot 1: Loss -----
    axes[0].plot(epochs, history["train_loss"],
                 label="Train", color="#4C72B0", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],
                 label="Val",   color="#DD8452", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (Cross-Entropy)")
    axes[0].set_title("Loss per epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.35)
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ----- Subplot 2: Accuracy -----
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]],
                 label="Train", color="#4C72B0", linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]],
                 label="Val",   color="#DD8452", linewidth=2, linestyle="--")
    axes[1].axhline(80, color="gray", linestyle=":", linewidth=1.2,
                    label="80% reference")   # Task 2 reference line
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy per epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.35)
    axes[1].set_ylim(0, 105)
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()

    if save:
        os.makedirs(figures_dir, exist_ok=True)
        path = os.path.join(figures_dir, f"curves_{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Figure saved: {path}")

    plt.show()


# ============================================================
# BLOCK 2 — MULTI-MODEL COMPARISON
# ============================================================

def plot_model_comparison(histories: dict,
                          save: bool = True,
                          figures_dir: str = FIGURES_DIR) -> None:
    """
    Plots val_loss and val_acc curves for multiple models
    in a single figure for direct comparison (Task 1).

    Args:
        histories   : dict of dicts — { model_name: history }
                      Example: {
                          "LeNet5"    : history_lenet,
                          "LeNet5+BN" : history_lenet_bn,
                          "VGG11"     : history_vgg,
                          "VGG11+BN"  : history_vgg_bn
                      }
        save        : if True, saves the figure
        figures_dir : destination folder
    """

    model_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison — Val Loss and Val Accuracy",
                 fontsize=13, fontweight="bold")

    for idx, (name, history) in enumerate(histories.items()):
        color  = model_colors[idx % len(model_colors)]
        epochs = range(1, len(history["val_loss"]) + 1)

        axes[0].plot(epochs, history["val_loss"],
                     label=name, color=color, linewidth=2)
        axes[1].plot(epochs, [a * 100 for a in history["val_acc"]],
                     label=name, color=color, linewidth=2)

    # Loss axis
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val Loss")
    axes[0].set_title("Validation Loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.35)
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Accuracy axis
    axes[1].axhline(80, color="gray", linestyle=":", linewidth=1.2,
                    label="80% reference")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Accuracy (%)")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.35)
    axes[1].set_ylim(0, 105)
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()

    if save:
        os.makedirs(figures_dir, exist_ok=True)
        path = os.path.join(figures_dir, "model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Figure saved: {path}")

    plt.show()


def summary_table(histories: dict,
                  epoch_times: dict = None) -> None:
    """
    Prints a comparative summary table of trained models.
    Useful for the Task 1 report.

    Args:
        histories   : dict { model_name: history }
        epoch_times : dict { model_name: float (seconds) }
                      If None, computed from history.
    """

    print("\n" + "=" * 75)
    print(f"  {'MODEL':<20} {'Val Acc':>9} {'Train Acc':>10} "
          f"{'Val Loss':>9} {'Best Ep':>8} {'t/epoch':>9}")
    print("=" * 75)

    for name, h in histories.items():
        best_acc   = h["best_val_acc"]
        best_epoch = h["best_epoch"]
        val_loss   = h["val_loss"][best_epoch - 1]
        train_acc  = h["train_acc"][best_epoch - 1]

        if epoch_times and name in epoch_times:
            t = epoch_times[name]
        else:
            t = np.mean(h["epoch_time"])

        print(f"  {name:<20} {best_acc:>8.2%} {train_acc:>9.2%} "
              f"{val_loss:>9.4f} {best_epoch:>8} {t:>7.1f}s")

    print("=" * 75 + "\n")


# ============================================================
# BLOCK 3 — CONFUSION MATRIX
# ============================================================

def get_predictions(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device = DEVICE) -> tuple:
    """
    Collects predictions and ground-truth labels over a DataLoader.

    Args:
        model  : trained model in eval mode
        loader : DataLoader (typically the test set)
        device : compute device

    Returns:
        (y_true, y_pred) — numpy arrays with class indices
    """

    model.eval()
    model = model.to(device)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list = CLASSES,
                          model_name: str = "Best Model",
                          save: bool = True,
                          figures_dir: str = FIGURES_DIR) -> None:
    """
    Plots the normalized confusion matrix of the model.

    Args:
        y_true       : array of ground-truth labels
        y_pred       : array of model predictions
        class_names  : list of class names
        model_name   : name for the title and output file
        save         : if True, saves the figure
        figures_dir  : destination folder
    """

    num_classes = len(class_names)

    # Build confusion matrix manually (no sklearn)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1

    # Normalize by row (per-class recall)
    matrix_norm = matrix.astype(float)
    for i in range(num_classes):
        row_total = matrix[i].sum()
        if row_total > 0:
            matrix_norm[i] = matrix[i] / row_total

    # Global accuracy
    global_acc = np.diag(matrix).sum() / matrix.sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {model_name}  "
                 f"(Test Accuracy: {global_acc:.2%})",
                 fontsize=13, fontweight="bold")

    # ----- Subplot 1: Absolute counts -----
    sns.heatmap(matrix,
                annot=True, fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[0],
                linewidths=0.5,
                cbar_kws={"label": "Count"})
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Ground truth")
    axes[0].set_title("Absolute counts")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].tick_params(axis="y", rotation=0)

    # ----- Subplot 2: Normalized (per-class recall) -----
    sns.heatmap(matrix_norm,
                annot=True, fmt=".2f",
                cmap="Blues", vmin=0, vmax=1,
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[1],
                linewidths=0.5,
                cbar_kws={"label": "Proportion"})
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Ground truth")
    axes[1].set_title("Normalized by class (Recall)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()

    if save:
        os.makedirs(figures_dir, exist_ok=True)
        filename = model_name.replace(" ", "_").lower()
        path = os.path.join(figures_dir, f"confusion_{filename}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Figure saved: {path}")

    plt.show()


# ============================================================
# BLOCK 4 — BATCH NORMALIZATION ANALYSIS
# ============================================================

def epoch_for_accuracy(history: dict,
                       threshold: float = 0.80,
                       split: str = "val") -> int:
    """
    Returns the first epoch at which the model reaches the given
    accuracy threshold. Returns -1 if never reached.

    Args:
        history   : training history dict
        threshold : target accuracy (default: 0.80 = 80%)
        split     : 'val' or 'train'

    Returns:
        epoch (int) at which the threshold is first reached, or -1
    """

    key = f"{split}_acc"
    assert key in history, f"Key '{key}' not found in history"

    for epoch, acc in enumerate(history[key], start=1):
        if acc >= threshold:
            return epoch

    return -1   # Threshold never reached


def plot_bn_convergence(history_no_bn: dict,
                        history_bn: dict,
                        architecture: str = "LeNet-5",
                        acc_threshold: float = 0.80,
                        save: bool = True,
                        figures_dir: str = FIGURES_DIR) -> None:
    """
    Plots the convergence comparison between an architecture
    with and without Batch Normalization. Designed for Task 2.

    Includes:
      - Train and val loss curves (left subplot)
      - Val accuracy curves with convergence markers (right subplot)

    Args:
        history_no_bn : history of the model without BN
        history_bn    : history of the model with BN
        architecture  : base name (e.g. 'LeNet-5', 'VGG-11')
        acc_threshold : accuracy reference line (default: 0.80)
        save          : if True, saves the figure
        figures_dir   : destination folder
    """

    epochs_no_bn = range(1, len(history_no_bn["val_loss"]) + 1)
    epochs_bn    = range(1, len(history_bn["val_loss"]) + 1)

    # Compute convergence epochs
    ep_no_bn = epoch_for_accuracy(history_no_bn, acc_threshold)
    ep_bn    = epoch_for_accuracy(history_bn,    acc_threshold)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Effect of Batch Normalization — {architecture}\n"
        f"(Ioffe & Szegedy, 2015 — Internal Covariate Shift)",
        fontsize=12, fontweight="bold"
    )

    # ----- Subplot 1: Loss curves -----
    axes[0].plot(epochs_no_bn, history_no_bn["train_loss"],
                 color="#4C72B0", linewidth=2,
                 linestyle="-",  label=f"{architecture} — Train")
    axes[0].plot(epochs_no_bn, history_no_bn["val_loss"],
                 color="#4C72B0", linewidth=2,
                 linestyle="--", label=f"{architecture} — Val")
    axes[0].plot(epochs_bn, history_bn["train_loss"],
                 color="#DD8452", linewidth=2,
                 linestyle="-",  label=f"{architecture}+BN — Train")
    axes[0].plot(epochs_bn, history_bn["val_loss"],
                 color="#DD8452", linewidth=2,
                 linestyle="--", label=f"{architecture}+BN — Val")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (Cross-Entropy)")
    axes[0].set_title("Loss Curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.35)
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ----- Subplot 2: Val Accuracy + convergence markers -----
    axes[1].plot(epochs_no_bn,
                 [a * 100 for a in history_no_bn["val_acc"]],
                 color="#4C72B0", linewidth=2,
                 label=f"{architecture}")
    axes[1].plot(epochs_bn,
                 [a * 100 for a in history_bn["val_acc"]],
                 color="#DD8452", linewidth=2,
                 label=f"{architecture}+BN")

    # Reference line at threshold
    axes[1].axhline(acc_threshold * 100, color="gray",
                    linestyle=":", linewidth=1.5,
                    label=f"{acc_threshold*100:.0f}% reference")

    # Convergence markers
    if ep_no_bn > 0:
        axes[1].axvline(ep_no_bn, color="#4C72B0", linestyle=":",
                        linewidth=1.2, alpha=0.7)
        axes[1].annotate(f"No BN\nep. {ep_no_bn}",
                         xy=(ep_no_bn, acc_threshold * 100),
                         xytext=(ep_no_bn + 0.5, acc_threshold * 100 - 8),
                         fontsize=8, color="#4C72B0")

    if ep_bn > 0:
        axes[1].axvline(ep_bn, color="#DD8452", linestyle=":",
                        linewidth=1.2, alpha=0.7)
        axes[1].annotate(f"With BN\nep. {ep_bn}",
                         xy=(ep_bn, acc_threshold * 100),
                         xytext=(ep_bn + 0.5, acc_threshold * 100 + 2),
                         fontsize=8, color="#DD8452")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Accuracy (%)")
    axes[1].set_title("Val Accuracy Convergence")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.35)
    axes[1].set_ylim(0, 105)
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Convergence summary text box
    summary = (
        f"No BN  → {acc_threshold*100:.0f}% at epoch: "
        f"{'Never' if ep_no_bn == -1 else ep_no_bn}\n"
        f"With BN → {acc_threshold*100:.0f}% at epoch: "
        f"{'Never' if ep_bn == -1 else ep_bn}"
    )
    axes[1].text(0.98, 0.05, summary,
                 transform=axes[1].transAxes,
                 fontsize=8, verticalalignment="bottom",
                 horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if save:
        os.makedirs(figures_dir, exist_ok=True)
        filename = architecture.replace("-", "").replace(" ", "_").lower()
        path = os.path.join(figures_dir, f"bn_convergence_{filename}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Figure saved: {path}")

    plt.show()

    # Console summary
    print(f"\nConvergence summary at {acc_threshold*100:.0f}%:")
    print(f"   {architecture}     → epoch {ep_no_bn if ep_no_bn > 0 else 'Never reached'}")
    print(f"   {architecture}+BN  → epoch {ep_bn    if ep_bn    > 0 else 'Never reached'}")


# ============================================================
# BLOCK 5 — TEST SET PREDICTIONS AND METRICS
# ============================================================

def classification_report(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           class_names: list = CLASSES) -> None:
    """
    Prints precision, recall and F1-score per class,
    computed manually without sklearn.

    Args:
        y_true       : array of ground-truth labels
        y_pred       : array of predictions
        class_names  : list of class names
    """

    num_classes = len(class_names)

    print("\n" + "=" * 60)
    print(f"  {'CLASS':<15} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>10}")
    print("=" * 60)

    precisions, recalls, f1s = [], [], []

    for i, cls in enumerate(class_names):
        tp = np.sum((y_true == i) & (y_pred == i))   # True positives
        fp = np.sum((y_true != i) & (y_pred == i))   # False positives
        fn = np.sum((y_true == i) & (y_pred != i))   # False negatives
        support = np.sum(y_true == i)                 # Total real samples for class

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"  {cls:<15} {precision:>10.3f} {recall:>8.3f} "
              f"{f1:>8.3f} {support:>10}")

    # Macro average
    print("-" * 60)
    print(f"  {'Macro avg':<15} {np.mean(precisions):>10.3f} "
          f"{np.mean(recalls):>8.3f} {np.mean(f1s):>8.3f} "
          f"{len(y_true):>10}")

    acc = np.sum(y_true == y_pred) / len(y_true)
    print(f"\n  Global accuracy: {acc:.4f} ({acc:.2%})")
    print("=" * 60 + "\n")


# ============================================================
# QUICK VERIFICATION (run directly: python utils.py)
# ============================================================

if __name__ == "__main__":

    print("\nVerifying utils.py with synthetic data...\n")

    np.random.seed(42)
    N = 200

    # Simulate training histories
    def synthetic_history(n_epochs=15, seed=0):
        np.random.seed(seed)
        base_loss = np.linspace(1.4, 0.4, n_epochs)
        base_acc  = np.linspace(0.30, 0.88, n_epochs)
        noise     = np.random.normal(0, 0.02, n_epochs)
        val_acc   = np.clip(base_acc + noise, 0, 1).tolist()
        best_idx  = int(np.argmax(val_acc))
        return {
            "train_loss"   : (base_loss - 0.05 + noise).tolist(),
            "val_loss"     : (base_loss + noise).tolist(),
            "train_acc"    : np.clip(base_acc + 0.02, 0, 1).tolist(),
            "val_acc"      : val_acc,
            "epoch_time"   : [2.5] * n_epochs,
            "best_val_acc" : float(max(val_acc)),
            "best_epoch"   : best_idx + 1
        }

    h_lenet    = synthetic_history(seed=0)
    h_lenet_bn = synthetic_history(seed=1)
    h_vgg      = synthetic_history(seed=2)
    h_vgg_bn   = synthetic_history(seed=3)

    # ----- Test: individual curves -----
    plot_curves(h_lenet, name="LeNet5_test", save=False)
    print("✓ plot_curves OK")

    # ----- Test: model comparison -----
    histories = {
        "LeNet5"    : h_lenet,
        "LeNet5+BN" : h_lenet_bn,
        "VGG11"     : h_vgg,
        "VGG11+BN"  : h_vgg_bn
    }
    plot_model_comparison(histories, save=False)
    print("✓ plot_model_comparison OK")

    # ----- Test: summary table -----
    summary_table(histories)
    print("✓ summary_table OK")

    # ----- Test: BN convergence -----
    plot_bn_convergence(h_lenet, h_lenet_bn,
                        architecture="LeNet-5",
                        save=False)
    print("✓ plot_bn_convergence OK")

    # ----- Test: confusion matrix -----
    y_true = np.random.randint(0, 4, N)
    y_pred = np.where(np.random.rand(N) > 0.25, y_true,
                      np.random.randint(0, 4, N))

    plot_confusion_matrix(y_true, y_pred,
                          model_name="Model_test",
                          save=False)
    print("✓ plot_confusion_matrix OK")

    # ----- Test: classification report -----
    classification_report(y_true, y_pred)
    print("✓ classification_report OK")

    # ----- Test: epoch for accuracy -----
    ep = epoch_for_accuracy(h_lenet, threshold=0.80)
    print(f"✓ epoch_for_accuracy OK — reaches 80% at epoch: {ep}")

    print("\n✓ utils.py verified successfully.\n")
