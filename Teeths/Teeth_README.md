# 🦷 Teeth Health Classification

A binary image classification project that detects whether teeth are **Healthy** or **Unhealthy** using a custom-built deep CNN with residual and attention mechanisms.

---

## 📌 Project Overview

This project trains models to classify dental images as either healthy or unhealthy. Two different approaches were implemented and compared:

1. **Custom CNN** (`teeth.py`) — A hand-crafted deep architecture with Residual Blocks + Squeeze-and-Excitation (SE) Attention
2. **EfficientNetB0 Transfer Learning** (`teethwith.py`) — Fine-tuned pretrained model

| | |
|---|---|
| ![Healthy Teeth](sample_images/healthy.jpg) | ![Unhealthy Teeth](sample_images/unhealthy.jpg) |
| *Healthy* | *Unhealthy* |

---

## 🧠 Model Architectures

### 1. Custom CNN with Residual + SE Attention (`teeth.py`)

A fully custom architecture inspired by ResNet with added channel attention:

```
Input (224×224×3)
    ↓
STEM: Conv(32) → Conv(64) → MaxPool
    ↓
Stage 1: ResidualBlock(64) × 2
    ↓
Stage 2: ResidualBlock(128) × 3
    ↓
Stage 3: ResidualBlock(256) × 3
    ↓
Stage 4: ResidualBlock(512) × 2
    ↓
GlobalAvgPool + GlobalMaxPool → Concatenate
    ↓
Dense(512) → BN → ReLU → Dropout(0.5)
    ↓
Dense(256) → BN → ReLU → Dropout(0.3)
    ↓
Dense(1, sigmoid)  →  Healthy / Unhealthy
```

Each **Residual Block** contains:
- 2× Conv2D + BatchNorm + ReLU
- **Squeeze-and-Excitation (SE) Block** — channel-wise attention
- Skip connection

### 2. EfficientNetB0 Transfer Learning (`teethwith.py`)

```
EfficientNetB0 (trainable, pretrained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) → Dropout(0.5)
    ↓
Dense(1, sigmoid)  →  Healthy / Unhealthy
```

---

## 🛠️ Data Pipeline

### Dataset Structure

```
├── healthy/
│   ├── img1.jpg
│   └── ...
└── unhealthy/
    ├── img1.jpg
    └── ...
```

### Preprocessing & Augmentation

| Augmentation | Value |
|---|---|
| Rotation | ±25° |
| Width/Height Shift | 20% |
| Shear | 15% |
| Zoom | 20% |
| Horizontal Flip | ✅ |
| Brightness | [0.8, 1.2] |

- **Train/Test Split:** 85% / 15% (stratified)
- **Class Weights:** Computed automatically to handle class imbalance

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install tensorflow scikit-learn pandas matplotlib seaborn
```

### 2. Prepare Dataset

Place your images in `healthy/` and `unhealthy/` folders.

### 3. Train — Custom CNN

```bash
python teeth.py
```

Saves best model as `best_custom_cnn.keras`

### 4. Train — EfficientNetB0

```bash
python teethwith.py
```

### 5. Test on New Images

```bash
python test.py
```

Edit `test.py` to specify your image paths.

---

## 📁 File Structure

```
Teeth/
├── teeth.py              # Custom CNN training script
├── teethwith.py          # EfficientNetB0 training script
├── test.py               # Quick inference on new images
├── healthy/              # Healthy teeth images
├── unhealthy/            # Unhealthy teeth images
└── best_custom_cnn.keras # Best saved model (not included – see note)
```

> ⚠️ **Note:** The trained model file (`best_custom_cnn.keras`) is not included in this repository due to GitHub's file size limit (154 MB). You can reproduce it by running `teeth.py`.

---

## ⚙️ Training Configuration

| Parameter | Custom CNN | EfficientNetB0 |
|---|---|---|
| Image Size | 224×224 | 224×224 |
| Batch Size | 32 | 32 |
| Max Epochs | 50 | 10 |
| Optimizer | Adam | Adam |
| Learning Rate | Cosine Annealing (warmup) | 0.0001 |
| Early Stopping | patience=12 (val_accuracy) | patience=3 (val_loss) |

---

## 📊 Training Callbacks (Custom CNN)

- **EarlyStopping** — Monitors `val_accuracy`, patience=12, restores best weights
- **ReduceLROnPlateau** — Halves LR if `val_loss` stalls for 4 epochs
- **ModelCheckpoint** — Saves best model based on `val_accuracy`
- **LearningRateScheduler** — Cosine annealing with 5-epoch linear warmup

---

## 🔍 Quick Inference

```python
from test import predict_teeth

result = predict_teeth("your_image.jpg")
print(result)
# Output: "Healthy (Confidence: 94.72%)"
#      or "Unhealthy (Confidence: 88.31%)"
```
