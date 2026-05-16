# ============================================================
# models.py — CNN Architectures
# Practical Assignment Week 6: CNN Comparison, BN and Transfer Learning
# Dataset: Blood Cell Image Dataset (4 classes)
# Framework: PyTorch
# ============================================================
#
# Contents:
#   BLOCK 1 — Adapted LeNet-5 (with and without Batch Normalization)
#   BLOCK 2 — Simplified VGG-11 (with and without Batch Normalization)
#   BLOCK 3 — ResNet-18 for Transfer Learning (3 strategies)
#   BLOCK 4 — Helper function: parameter summary
#
# Usage:
#   from src.models import LeNet5, LeNet5BN, VGGSimple, VGGSimpleBN
#   from src.models import get_resnet18, count_parameters
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

NUM_CLASSES  = 4     # EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL
INPUT_SIZE   = 64    # Input resolution for LeNet-5 and VGG-11
INPUT_SIZE_TL = 224  # Input resolution for Transfer Learning


# ============================================================
# BLOCK 1 — ADAPTED LeNet-5
# ============================================================
#
# Original architecture (LeCun et al., 1998) adapted for:
#   - RGB input of 64×64×3 (instead of grayscale 32×32)
#   - 4 output classes (instead of 10 digits)
#   - ReLU activation (instead of original tanh)
#
# Dimension flow:
#   Input : (B, 3, 64, 64)
#   Conv1 : (B, 6, 60, 60)  → Pool → (B,  6, 30, 30)
#   Conv2 : (B, 16, 26, 26) → Pool → (B, 16, 13, 13)
#   Flatten: 16 × 13 × 13 = 2704
#   FC1   : 2704 → 120
#   FC2   : 120  → 84
#   FC3   : 84   → 4
# ============================================================

class LeNet5(nn.Module):
    """
    LeNet-5 adapted to the blood cell dataset.
    Without Batch Normalization — baseline version.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(LeNet5, self).__init__()

        # ----- CONVOLUTIONAL BLOCK 1 -----
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,   # RGB input (3 channels)
                      out_channels=6,  # 6 filters
                      kernel_size=5),  # Kernel 5×5 → output: 60×60
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,  # 2×2 window
                         stride=2)       # → output: 30×30
        )

        # ----- CONVOLUTIONAL BLOCK 2 -----
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=6,    # Input: 6 feature maps
                      out_channels=16,  # 16 filters
                      kernel_size=5),   # Kernel 5×5 → output: 26×26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)      # → output: 13×13
        )

        # ----- FULLY CONNECTED CLASSIFIER -----
        self.classifier = nn.Sequential(
            nn.Flatten(),              # 16 × 13 × 13 = 2704 neurons

            nn.Linear(16 * 13 * 13,   # FC1: 2704 → 120
                      120),
            nn.ReLU(),

            nn.Linear(120, 84),        # FC2: 120 → 84
            nn.ReLU(),

            nn.Linear(84, num_classes) # FC3: 84 → 4 (raw logits, no softmax)
        )

    def forward(self, x):
        x = self.conv_block1(x)    # First conv-pool block
        x = self.conv_block2(x)    # Second conv-pool block
        x = self.classifier(x)     # FC classification
        return x


# ------------------------------------------------------------

class LeNet5BN(nn.Module):
    """
    LeNet-5 adapted with Batch Normalization.
    BN is applied after each convolutional layer and before ReLU,
    following the recommendation of Ioffe & Szegedy (2015).
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(LeNet5BN, self).__init__()

        # ----- CONVOLUTIONAL BLOCK 1 + BN -----
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=6,
                      kernel_size=5),
            nn.BatchNorm2d(6),   # BN sobre los 6 mapas de características
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ----- CONVOLUTIONAL BLOCK 2 + BN -----
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5),
            nn.BatchNorm2d(16),  # BN sobre los 16 mapas de características
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ----- FULLY CONNECTED CLASSIFIER -----
        # Misma estructura que LeNet5 base (BN no se aplica en FC aquí)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


# ============================================================
# BLOCK 2 — SIMPLIFIED VGG-11
# ============================================================
#
# Original architecture (Simonyan & Zisserman, 2014) adapted for:
#   - Filters halved to be trainable on CPU/Colab
#   - RGB input of 64×64×3
#   - 4 output classes
#
# Original VGG-11 filters: [64, 128, 256, 256, 512, 512, 512, 512]
# Simplified filters:      [32,  64, 128, 128, 256, 256, 256, 256]
#
# Dimension flow (64×64 input):
#   Block 1: (B,  32, 64, 64) → Pool → (B,  32, 32, 32)
#   Block 2: (B,  64, 32, 32) → Pool → (B,  64, 16, 16)
#   Block 3: (B, 128, 16, 16) → Pool → (B, 128,  8,  8)
#   Block 4: (B, 256,  8,  8) → Pool → (B, 256,  4,  4)
#   Block 5: (B, 256,  4,  4) → Pool → (B, 256,  2,  2)
#   Flatten : 256 × 2 × 2 = 1024
#   FC1: 1024 → 512
#   FC2:  512 → 512
#   FC3:  512 → 4
# ============================================================

def _vgg_block(in_ch: int, out_ch: int,
               n_conv: int, batch_norm: bool) -> nn.Sequential:
    """
    Builds a VGG block: N conv3×3 layers + ReLU (+ optional BN) + MaxPool.

    Args:
        in_ch      : input channels
        out_ch     : output channels (filters)
        n_conv     : number of convolutional layers in the block
        batch_norm : if True, adds BatchNorm2d after each Conv2d
    """
    layers = []
    for i in range(n_conv):
        layers.append(
            nn.Conv2d(in_channels=in_ch if i == 0 else out_ch,
                      out_channels=out_ch,
                      kernel_size=3,
                      padding=1)    # padding=1 preserves spatial size
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Halves spatial size
    return nn.Sequential(*layers)


class VGGSimple(nn.Module):
    """
    Simplified VGG-11 (filters halved) without Batch Normalization.
    Baseline architecture for comparison in Task 1.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, batch_norm: bool = False):
        super(VGGSimple, self).__init__()

        # ----- CONVOLUTIONAL BLOCKS -----
        # Config: (output_filters, n_conv_per_block)
        # VGG-11 has 8 conv layers distributed across 5 blocks: 1,1,2,2,2
        self.features = nn.Sequential(
            _vgg_block(3,    32,  n_conv=1, batch_norm=batch_norm),   # Block 1
            _vgg_block(32,   64,  n_conv=1, batch_norm=batch_norm),   # Block 2
            _vgg_block(64,  128,  n_conv=2, batch_norm=batch_norm),   # Block 3
            _vgg_block(128, 256,  n_conv=2, batch_norm=batch_norm),   # Block 4
            _vgg_block(256, 256,  n_conv=2, batch_norm=batch_norm),   # Block 5
        )

        # ----- CLASIFICADOR -----
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256 * 2 * 2, 512),  # FC1: 1024 → 512
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),             # Dropout for regularization

            nn.Linear(512, 512),           # FC2: 512 → 512
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(512, num_classes)    # FC3: 512 → 4
        )

    def forward(self, x):
        x = self.features(x)       # Feature extraction
        x = self.classifier(x)     # Classification
        return x


# ------------------------------------------------------------

class VGGSimpleBN(VGGSimple):
    """
    Simplified VGG-11 WITH Batch Normalization.
    Inherits from VGGSimple passing batch_norm=True.
    Enables direct comparison with VGGSimple in Task 2.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(VGGSimpleBN, self).__init__(num_classes=num_classes,
                                          batch_norm=True)


# ============================================================
# BLOCK 3 — ResNet-18 FOR TRANSFER LEARNING
# ============================================================
#
# Transfer Learning strategies (Task 3):
#
#   'feature_extraction'  → Freeze ALL conv layers.
#                           Only train the final FC layer.
#
#   'partial_fine_tuning' → Freeze layer1 and layer2.
#                           Train layer3, layer4 and FC.
#
#   'full_fine_tuning'    → Unfreeze the entire model.
#                           Train with very small lr (~1e-5).
#
# The function returns the configured model + parameters to optimize.
# ============================================================

def get_resnet18(strategy: str,
                 num_classes: int = NUM_CLASSES) -> tuple:
    """
    Loads ResNet-18 pretrained on ImageNet and configures it
    according to the specified Transfer Learning strategy.

    Args:
        strategy   : 'feature_extraction' | 'partial_fine_tuning' | 'full_fine_tuning'
        num_classes: number of output classes (default: 4)

    Returns:
        model      : configured model (nn.Module)
        params     : list of parameters to pass to the optimizer

    Example usage:
        model, params = get_resnet18('partial_fine_tuning')
        optimizer = torch.optim.Adam(params, lr=1e-4)
    """

    valid_strategies = [
        "feature_extraction",
        "partial_fine_tuning",
        "full_fine_tuning"
    ]
    assert strategy in valid_strategies, (
        f"Strategy '{strategy}' is not valid. "
        f"Options: {valid_strategies}"
    )

    # Load ResNet-18 with ImageNet pretrained weights
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

    # ----------------------------------------------------------
    # STRATEGY 1: Feature Extraction
    # Freeze ALL parameters except the FC classifier
    # ----------------------------------------------------------
    if strategy == "feature_extraction":

        # Freeze all backbone parameters
        for param in model.parameters():
            param.requires_grad = False

        # Replace final FC: 512 → num_classes (unfrozen by default)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Only FC is passed to the optimizer
        params = model.fc.parameters()

    # ----------------------------------------------------------
    # STRATEGY 2: Partial Fine-tuning
    # Freeze layer1 and layer2. Train layer3, layer4 and FC.
    # ----------------------------------------------------------
    elif strategy == "partial_fine_tuning":

        # First freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze layer3, layer4 and FC
        for block in [model.layer3, model.layer4]:
            for param in block.parameters():
                param.requires_grad = True

        # Replace and unfreeze FC
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Trainable parameters: layer3 + layer4 + FC
        params = filter(lambda p: p.requires_grad, model.parameters())

    # ----------------------------------------------------------
    # STRATEGY 3: Full Fine-tuning
    # Unfreeze the entire model, use very small lr
    # ----------------------------------------------------------
    elif strategy == "full_fine_tuning":

        # All parameters trainable
        for param in model.parameters():
            param.requires_grad = True

        # Replace final FC
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # All parameters passed to the optimizer
        params = model.parameters()

    return model, list(params)


# ============================================================
# BLOCK 4 — HELPER FUNCTION: PARAMETER SUMMARY
# ============================================================

def count_parameters(model: nn.Module,
                     trainable_only: bool = False) -> dict:
    """
    Counts the parameters of a PyTorch model.

    Args:
        model          : nn.Module model
        trainable_only : if True, counts only parameters with grad

    Returns:
        dict with 'total', 'trainable', 'frozen'

    Example usage:
        info = count_parameters(model)
        print(f"Total parameters: {info['total']:,}")
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    return {
        "total"     : total,
        "trainable" : trainable,
        "frozen"    : frozen
    }


def print_summary(model: nn.Module, name: str = "Model") -> None:
    """
    Prints a formatted parameter summary.

    Args:
        model : nn.Module model
        name  : name of the model to display
    """
    info = count_parameters(model)
    print(f"{'=' * 45}")
    print(f"  {name}")
    print(f"{'=' * 45}")
    print(f"  Total parameters    : {info['total']:>12,}")
    print(f"  Trainable parameters: {info['trainable']:>12,}")
    print(f"  Frozen parameters   : {info['frozen']:>12,}")
    print(f"{'=' * 45}")


# ============================================================
# QUICK VERIFICATION (run directly: python models.py)
# ============================================================

if __name__ == "__main__":

    print("\nVerifying architectures with 64×64×3 input...\n")

    test_batch = torch.randn(4, 3, 64, 64)     # Batch of 4 images

    # --- LeNet-5 ---
    model = LeNet5()
    output = model(test_batch)
    assert output.shape == (4, NUM_CLASSES), "Error in LeNet5"
    print_summary(model, "LeNet-5 (no BN)")

    # --- LeNet-5 + BN ---
    model = LeNet5BN()
    output = model(test_batch)
    assert output.shape == (4, NUM_CLASSES), "Error in LeNet5BN"
    print_summary(model, "LeNet-5 (with BN)")

    # --- Simplified VGG-11 ---
    model = VGGSimple()
    output = model(test_batch)
    assert output.shape == (4, NUM_CLASSES), "Error in VGGSimple"
    print_summary(model, "VGG-11 Simple (no BN)")

    # --- Simplified VGG-11 + BN ---
    model = VGGSimpleBN()
    output = model(test_batch)
    assert output.shape == (4, NUM_CLASSES), "Error in VGGSimpleBN"
    print_summary(model, "VGG-11 Simple (with BN)")

    print("\nVerifying ResNet-18 with 224×224×3 input...\n")

    tl_batch = torch.randn(4, 3, 224, 224)

    for strategy in ["feature_extraction", "partial_fine_tuning", "full_fine_tuning"]:
        model, params = get_resnet18(strategy)
        output = model(tl_batch)
        assert output.shape == (4, NUM_CLASSES), f"❌ Error in ResNet18 ({strategy})"
        print_summary(model, f"ResNet-18 [{strategy}]")

    print("\n✓ All models verified successfully.\n")
