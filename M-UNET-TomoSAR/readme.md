# Forest Height Estimation with Deep Learning M-UNET and TomoSAR

This repository contains the implementation of a proposed deep learning model (CNN/M-Unet) for **forest canopy height (CHM)** and **ground height (DTM)** estimation using **Polarimetric TomoSAR data**.  
The model is evaluated against **traditional spectral estimation methods** (MUSIC and CAPON) and validated using **LiDAR ground truth labels**.

---

## Overview

- **Goal:** Estimate absolute forest canopy and ground heights from P-band TomoSAR data.
- **Data:** 
  - Complex covariance matrices (18 × 18).
  - 18 SAR intensity images.
  - LiDAR-derived CHM and DTM (absolute heights, reference to sea level, with calibration offset).
- **Methods Compared:**
  - **Deep Learning:**
    - CNN/M-Unet(proposed).
  - **Traditional:** 
    - MUSIC (multiple polarizations and average).
    - CAPON (multiple polarizations and average).
- **Evaluation Metrics:**
  - Root Mean Square Error (RMSE).
  - Mean Absolute Error (MAE).
  - Bias.

---

##  Methodology

1. **Input Features:**
   - Diagonal and first-row off-diagonal elements for each pixel of the covariance matrix.
   - Full set of 18 SAR images.
   - Using patches of mutual values pixels instead of single pixels.
   - Combined into a **70-channel feature stack**.

2. **Deep Learning Model:**
   - CNN/U-Net style architecture (M-UNET).
   - Predicts **absolute CHM and DTM** at voxel-level.
   - Training with cross-entropy (class-based).

3. **Traditional Methods:**
   - MUSIC and CAPON implemented from provided `.mat` files.
   - Averaged over polarizations (`hh`, `hv`, `vv`).

4. **Evaluation:**
   - Metrics computed vs. LiDAR GT for both **CHM** and **DTM**.
   - Also compare **relative canopy height** (CHM − DTM).

---

## Results

- **CNN/TSNN** achieves lower RMSE/MAE for CHM and DTM compared to MUSIC and CAPON.
- **Traditional methods** show larger bias, especially at canopy level.
- Error maps and transect plots included for visualization.

### Example Results

| Method   | Variable | RMSE (m) | MAE (m) | Bias (m) |
|----------|----------|----------|---------|----------|
| CNN      | CHM      | 1.08     | 1.00    | -1.00    |
| TSNN     | CHM      | 12.28    | 9.19    | -8.91    |
| MUSIC    | CHM      | 12.61    | 10.03   | -8.84    |
| CAPON    | CHM      | 7.65     | 5.89    | +1.67    |
| CNN      | DTM      | 0.58     | 0.50    | -0.50    |
| TSNN     | DTM      | 7.11     | 4.36    | -3.77    |
| MUSIC    | DTM      | 6.99     | 5.13    | +4.94    |
| CAPON    | DTM      | 4.11     | 3.01    | -0.97    |
| CNN      | Canopy   | 0.58     | 0.50    | -0.50    |
| TSNN     | Canopy   | 6.61     | 5.31    | -5.13    |
| MUSIC    | Canopy   | 16.08    | 14.03   | -13.78   |
| CAPON    | Canopy   | 8.63     | 6.61    | +2.64    |

---

## Repository Structure
```
├── data/ # Input TomoSAR covariance matrices & SAR images
├── TM/ # MUSIC and CAPON .mat results
├── models/ # CNN/TSNN architecture code
├── notebooks/ # Jupyter notebooks for training & evaluation
├── test/ # Output predictions, metrics, visualizations
├── utils/ # Helper functions
└── README.md # Project description
```

##  Usage
### Training
You can download and run the notebook, but due to the privacy of the datasets, you can only apply this concept once you have your own data.
```python
model, splits, history = train_and_save(
    dataset,
    out_dir=out_dir,
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    device=device,
    val_ratio=0.1,
    test_ratio=0.1,
    base=32,
    patience=6,
    save_every_epoch=True
)
train_ds, val_ds, test_ds = splits

```

## Visualization

After training, you can visualize predicted vs true maps:

```python
x = np.arange(gt_chm_line.shape[0])  # horizontal pixel positions

plt.figure(figsize=(12, 5))

# --- CHM plot ---
plt.subplot(1, 2, 1)
plt.plot(x, gt_chm_line, label='LiDAR (GT)', linewidth=2, color='black')
plt.plot(x, cnn_chm_line, label='M-Unet', linestyle='--')
if music_chm_line is not None:
    plt.plot(x, music_chm_line, label='MUSIC', linestyle='-.')
if capon_chm_line is not None:
    plt.plot(x, capon_chm_line, label='CAPON', linestyle=':')
plt.title(f'Canopy Height Line Profile (Row {row})')
plt.xlabel('Pixel Position')
plt.ylabel('CHM (meters)')
plt.legend()
plt.grid(True)

# --- DTM plot ---
plt.subplot(1, 2, 2)
plt.plot(x, gt_dtm_line, label='LiDAR (GT)', linewidth=2, color='black')
plt.plot(x, cnn_dtm_line, label='M-Unet', linestyle='--')
if music_dtm_line is not None:
    plt.plot(x, music_dtm_line, label='MUSIC', linestyle='-.')
if capon_dtm_line is not None:
    plt.plot(x, capon_dtm_line, label='CAPON', linestyle=':')
plt.title(f'Terrain Height Line Profile (Row {row})')
plt.xlabel('Pixel Position')
plt.ylabel('DTM (meters)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"line_profiles_row_{row}.png"), dpi=150)
plt.show()

```
![CHM/ DTM predictions Transact line of M-Unet vs MUSIC vs CAPON compared to LiDar](https://github.com/user-attachments/assets/1ee9015a-5504-437b-8d88-ac9a1cf6258c)

---

## Visualizations of the predictions of CHM and DTM
### These are the CHM/DTM predictions of the two models, TSNN and my proposed model M-Unet, versus traditional methods like MUSIC and CAPON. The reference is the LIDAR labels.
![CHM Map](https://github.com/user-attachments/assets/028a1d67-4d50-43c3-906b-e748a925fdc9)

![DTM Map](https://github.com/user-attachments/assets/1dc80797-ca04-495f-bbae-7d5532c8355c)

