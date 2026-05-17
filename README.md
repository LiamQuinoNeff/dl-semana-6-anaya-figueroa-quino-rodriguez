# Blood Cell Classification — CNN Architectures
### PC2 · Introduction to Deep Learning · NRC 18522

**Peruvian University of Applied Sciences (UPC)**  
Faculty of Engineering · Computer Science Program

---

## Group Members

| Student Code | Full Name |
|---|---|
| U202210644 | Anaya Vadillo, Nathaly Eliane |
| U202220990 | Figueroa Larragán, Marsi Valeria |
| U20221E167 | Quino Neff, Liam Mikael |
| U202115571 | Rodriguez Matos, Loana Colleen |

**Professor:** Jairo Pinedo Taquía

---

## Project Overview

This project implements and compares classical Convolutional Neural Network (CNN) architectures for multi-class classification of blood cell microscopy images. Using the [Blood Cell Image Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) (4 classes: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL), the work covers four main topics:

- **Task 1** — From-scratch implementation of LeNet-5 and VGG-11 (with and without Batch Normalization)
- **Task 2** — Controlled analysis of the effect of Batch Normalization on convergence speed and training stability
- **Task 3** — Transfer Learning with pretrained ResNet-18 under three strategies: Feature Extraction, Partial Fine-tuning, and Full Fine-tuning
- **Task 4** — Technical report comparing all approaches (see `informe.pdf`)

**Framework:** PyTorch · **Dataset:** ~12,500 RGB images · **Input size:** 64×64 (CNN) / 224×224 (Transfer Learning)

---

## Repository Structure

```
dl-semana6-anaya-figueroa-quino-rodriguez/
├── README.md
├── data/
│   └── download_data.sh          # Kaggle dataset download script
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_lenet_vgg.ipynb        # Tasks 1 & 2: CNN training + BN analysis
│   └── 03_transfer.ipynb         # Task 3: Transfer Learning
├── src/
│   ├── models.py                 # LeNet-5, VGG-11, ResNet-18 architectures
│   ├── train.py                  # Reusable training loop + checkpoints
│   └── utils.py                  # Plotting and metric functions
├── results/
│   ├── figures/                  # All generated plots (saved automatically)
│   └── checkpoints/              # Best model weights per experiment (.pt)
└── informe.pdf
```

---

## Results Summary

| Model | Test Accuracy | Parameters | Avg. time/epoch |
|---|---|---|---|
| LeNet-5 | 63.57% | ~338K | 13.7s |
| LeNet-5 + BN | 69.16% | ~339K | 12.3s |
| VGG-11 Simple | 89.06% | ~3.1M | 12.4s |
| VGG-11 Simple + BN | 87.41% | ~3.1M | 13.2s |
| ResNet-18 Feature Ext. | 60.47% | ~2K trainable | 16.9s |
| ResNet-18 Partial FT | 80.30% | ~5.2M trainable | 16s |
| ResNet-18 Full FT | 85.32% | ~11.2M trainable | 16.7s |

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.5.0
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Run — Local

### 1. Clone the repository

```bash
git clone https://github.com/liamquinoneff/dl-semana6-anaya-figueroa-quino-rodriguez.git
cd dl-semana6-anaya-figueroa-quino-rodriguez
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

You need a Kaggle account and your API credentials (`kaggle.json`) configured.  
See [Kaggle API setup](https://github.com/Kaggle/kaggle-api#api-credentials) if needed.

```bash
bash data/download_data.sh
```

This will download and extract the dataset into `data/blood-cells/`.

### 5. Run the notebooks in order

```bash
# Option A — Jupyter Notebook
jupyter notebook

# Option B — JupyterLab
jupyter lab
```

Open and run the notebooks **in order**:

| Order | Notebook | Description |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | EDA — computes dataset statistics used by all other notebooks |
| 2 | `notebooks/02_lenet_vgg.ipynb` | Tasks 1 & 2 — requires `src/` modules |
| 3 | `notebooks/03_transfer.ipynb` | Task 3 — requires checkpoints from notebook 02 |

> **Note:** All figures are saved automatically to `results/figures/` and all best model checkpoints to `results/checkpoints/`. Make sure these folders exist or let the code create them on first run.

---

## How to Run — Google Colab

### 1. Open the notebook in Colab

Click one of the links below (or upload the `.ipynb` file manually to [colab.research.google.com](https://colab.research.google.com)):

- `01_eda.ipynb` → *Upload to Colab*
- `02_lenet_vgg.ipynb` → *Upload to Colab*
- `03_transfer.ipynb` → *Upload to Colab*

### 2. Enable GPU runtime

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

### 3. Clone the repository inside Colab

Add this cell at the top of any notebook before running:

```python
# Clone repo and set working directory
!git clone https://github.com/liamquinoneff/dl-semana6-anaya-figueroa-quino-rodriguez.git
%cd dl-semana6-anaya-figueroa-quino-rodriguez

# Install dependencies
!pip install -r requirements.txt -q
```

### 4. Download the dataset from Kaggle

```python
# Upload your kaggle.json when prompted
from google.colab import files
files.upload()   # select your kaggle.json file

# Move credentials to the expected location
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Run the download script
!bash data/download_data.sh
```

### 5. Run the notebooks in order

Run `01_eda.ipynb` first, then `02_lenet_vgg.ipynb`, then `03_transfer.ipynb`.

> **Tip:** To avoid losing checkpoints if the Colab session disconnects, mount Google Drive and update the `checkpoint_dir` path in the notebooks:

```python
from google.colab import drive
drive.mount('/content/drive')

# Then update in notebook config:
checkpoint_dir = "/content/drive/MyDrive/dl-semana6/checkpoints"
```

---

## Key Decisions

- **Seed:** All experiments use `SEED = 42` for full reproducibility (`torch`, `numpy`, `random`, CUDA)
- **Val split:** 15% of the training set is held out as validation across all notebooks
- **Checkpointing:** `save_best=True` saves only the epoch with the highest val accuracy
- **Normalization:** 64×64 models use dataset-specific Mean/Std computed in the EDA; Transfer Learning uses standard ImageNet Mean/Std
- **BN placement:** Following Ioffe & Szegedy (2015), BatchNorm is applied after Conv2d and before ReLU

---

## References

- LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS*.
- Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *ICLR 2015*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*.
