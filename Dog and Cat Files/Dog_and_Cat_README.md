# 🐾 Dog and Cat Classification

A binary image classification project that distinguishes between dogs and cats using **Transfer Learning with MobileNetV2**.

---

## 📌 Project Overview

This project uses the [Microsoft Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765) to train a deep learning model capable of classifying pet images as either a **cat** or a **dog**.

| | |
|---|---|
| ![Cat Example](sample_images/cat.jpg) | ![Dog Example](sample_images/dog.jpg) |
| *Cat* | *Dog* |

---

## 🧠 Model Architecture

The model is built on top of **MobileNetV2** pretrained on ImageNet, with a custom classification head:

```
MobileNetV2 (frozen, pretrained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(1, activation='sigmoid')  →  Cat / Dog
```

- **Input Size:** 224×224×3
- **Optimizer:** Adam (lr=0.0001)
- **Loss:** Binary Crossentropy
- **Metrics:** Accuracy

---

## 🛠️ Data Pipeline

- Dataset is loaded from the `PetImages/` directory using `image_dataset_from_directory`
- **80/20 Train-Validation split**
- Images are preprocessed using `MobileNetV2`'s `preprocess_input`
- Performance optimized with `.cache()`, `.shuffle()`, and `.prefetch()`

### Data Cleaning (`clean2.py`)

Before training, corrupted images are removed automatically:

```bash
python clean2.py
```

This script scans all images in `PetImages/`, tries to decode each one with TensorFlow, and deletes any file that raises an error.

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn
```

### 2. Prepare Dataset

Download the dataset and place it in a folder named `PetImages/` with the following structure:

```
PetImages/
├── Cat/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

### 3. Clean Corrupted Images

```bash
python clean2.py
```

### 4. Train the Model

```bash
python main.py
```

The best model will be saved as `best_model_mobilnet.h5`.

---

## 📁 File Structure

```
Dog and Cat Files/
├── main.py          # Main training script
├── clean2.py        # Corrupted image cleaner
├── PetImages/       # Dataset directory
│   ├── Cat/
│   └── Dog/
└── best_model.h5    # Saved best model (not included – see note below)
```

> ⚠️ **Note:** The trained model file (`best_model.h5`) is not included in this repository due to GitHub's file size limit (90 MB). You can reproduce it by running `main.py`.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 224×224 |
| Batch Size | 32 |
| Max Epochs | 10 |
| Early Stopping | patience=3 (val_loss) |
| Base Model | MobileNetV2 (frozen) |
| Learning Rate | 0.0001 |

---

## 📊 Training Callbacks

- **EarlyStopping** — Stops training if `val_loss` doesn't improve for 3 epochs, restores best weights
- **ModelCheckpoint** — Saves the best model based on `val_loss`
