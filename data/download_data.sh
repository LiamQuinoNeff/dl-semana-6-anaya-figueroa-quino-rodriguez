#!/bin/bash
# download_data.sh
# Descarga el Blood Cell Image Dataset desde Kaggle
# Requisito: tener configurado kaggle.json con tus credenciales

# ── 1. Verificar que kaggle CLI está instalado ──────────────────
if ! command -v kaggle &> /dev/null; then
    echo "Instalando kaggle CLI..."
    pip install kaggle -q
fi

# ── 2. Ubicar credenciales ──────────────────────────────────────
# En Colab: sube tu kaggle.json manualmente o monta Drive
# En local:  el archivo debe estar en ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: No se encontró ~/.kaggle/kaggle.json"
    echo "Descárgalo desde: https://www.kaggle.com/settings → API → Create New Token"
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json

# ── 3. Descargar y descomprimir ─────────────────────────────────
mkdir -p data/raw
kaggle datasets download -d paultimothymooney/blood-cells \
    --path data/raw \
    --unzip

echo "✓ Dataset descargado en data/raw/"