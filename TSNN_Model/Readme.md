# 🌲 TSNN: TomoSAR Neural Network for Canopy & Ground Height Estimation

**Author:** Sahar Mohamed  
**Internship:** July 2025 – September 2025, EO Analytics, Salzburg University, Austria  
**Supervisor:** Dr. Karima Hadj-Rabah

## Project overview
This repository contains the implementation of a **Two-Stream Neural Network (TSNN)** for estimating **canopy height (CHM)** and **ground height (DTM)** from **TomoSAR-derived features**.  

The project is built with **PyTorch Lightning** for structured training, reproducibility, and scalability.

---

##  Overview

The TSNN model is designed to predict:
- **CHM (Canopy Height Model)** → vegetation height above sea level (absolute height).
- **DTM (Digital Terrain Model)** → ground elevation above sea level (absolute height).

### Why Two-Stream?
- **Model C**: learns canopy heights (multi-class classification with 61 classes).
- **Model G**: learns ground heights (multi-class classification with 41 classes).  
- Predictions are trained jointly with **CrossEntropyLoss**, allowing the network to learn vegetation and terrain simultaneously.

---

## 📂 Repository Structure

```
.
├── train_data.npz          # Training dataset (features + labels)
├── val_data.npz            # Validation dataset
├── test_labeled_data.npz   # Test dataset (labeled)
├── tsnn_scaler.pkl         # Saved StandardScaler (fitted on training set)
├── TSNN.ipynb                # Core TSNN + Lightning module
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies

```

---

## Features

- Data handling with `LightningDataModule`.
- Standardization using `sklearn.StandardScaler`.
- Two-stream architecture (separate canopy & ground branches).
- Batch normalization + dropout regularization.
- Gradient clipping for stability.
- Checkpointing and early stopping.
- Automatic GPU/CPU handling.

---

## Installation

Clone the repo and install requirements:

```bash
git clone https://github.com/SaharAbdulallim/Deep-Learning-in-TomoSAR.git
cd  Deep-Learning-in-TomoSAR
pip install -r requirements.txt
```

---

## Data Format

Datasets (`.npz` files) should contain:
- `X` → feature matrix (samples × 52 features).
- `yc` → canopy height labels (class indices).
- `yg` → ground height labels (class indices).

Example to inspect a dataset:

```python
import numpy as np
data = np.load("train_data.npz")
print(data["X"].shape, data["yc"].shape, data["yg"].shape)
```

---

## Training

Run training with:

```python
import pytorch_lightning as pl
from TSNN import TSNNLitModule, HeightDataModule

pl.seed_everything(42)

data = HeightDataModule(
    train_npz='train_data.npz',
    val_npz='val_data.npz',
    test_npz='test_labeled_data.npz',
    batch_size=32
)

model = TSNNLitModule()

trainer = pl.Trainer(
    max_epochs=200,
    accelerator="auto",
    devices="auto",
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor='val/loss', save_top_k=1, mode='min', filename='best-tsnn'),
        pl.callbacks.EarlyStopping(monitor='val/loss', patience=20, mode='min')
    ]
)

trainer.fit(model, data)
```

---

## Visualization

After training, you can visualize predicted vs true maps:

```python
import matplotlib.pyplot as plt

# Example: visualize ground prediction
plt.imshow(pred_g.detach().cpu().numpy(), cmap='terrain', vmin=0, vmax=40)
plt.colorbar()
plt.title("Predicted DTM")
plt.show()
```
![CHM and DTM TSNN Predictions vs True Lidar CHM/DTM](https://github.com/user-attachments/assets/b21ca22b-2b28-4963-8970-49d22871a28a)

---

## Requirements

Example `requirements.txt`:

```
torch
pytorch-lightning
numpy
scikit-learn
joblib
matplotlib
panda
```

---

##  Future Work

- Extend to U-Net for voxel-wise segmentation and this in the other folder (M-Unet).
- Incorporate temporal TomoSAR stacks.
- Compare against M-Net architecture.


## Acknowledgements

This work was performed as an internship at EO Analytics, Salzburg University. Supervisor: Dr. Karima Hadj-Rabah.
