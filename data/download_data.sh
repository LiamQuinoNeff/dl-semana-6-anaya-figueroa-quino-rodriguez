#!/bin/bash
# download_data.sh
# Download the Blood Cell Image Dataset from Kaggle
# Requirement: kaggle.json must be configured with your credentials

# ── 1. Check that the kaggle CLI is installed ──────────────────
if ! command -v kaggle &> /dev/null; then
    echo "Installing kaggle CLI..."
    pip install kaggle -q
fi

# ── 2. Locate credentials ──────────────────────────────────────
# In Colab: upload kaggle.json manually or mount Drive
# Locally: the file must be at ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: ~/.kaggle/kaggle.json was not found"
    echo "Download it from: https://www.kaggle.com/settings → API → Create New Token"
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json

# ── 3. Download and extract ────────────────────────────────────
mkdir -p data/raw
kaggle datasets download -d paultimothymooney/blood-cells \
    --path data/raw \
    --unzip

echo "✓ Dataset downloaded to data/raw/"