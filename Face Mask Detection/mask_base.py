# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                     Dropout, BatchNormalization, GlobalAveragePooling2D,
                                     Input)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# GPU Memory Growth settings;
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# CONFIGURATION;
IMG_SIZE = 128  
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DATA_DIR = "data"

# Data Loaded;

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2  
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print(f"\n[INFO] Classes: {class_names}")
print(f"[INFO] Training samples: {train_generator.samples}")
print(f"[INFO] Validation samples: {val_generator.samples}")

# MODEL ARCHITECTURE - MobileNetV2 Transfer Learning

def create_mobilenetv2_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=True)
    
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    return model

model = create_mobilenetv2_model()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
print(f"\n[INFO] Trainable parameters: {trainable_params:,}")
print(f"[INFO] Non-trainable parameters: {non_trainable_params:,}")

# Callbacks;
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    
    ModelCheckpoint(
        'best_mask_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Training;

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# EVALUATION;

val_generator.reset()
val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
print(f"\n[RESULT] Validation Loss: {val_loss:.4f}")
print(f"[RESULT] Validation Accuracy: {val_accuracy*100:.2f}%")

val_generator.reset()
predictions = model.predict(val_generator, verbose=0)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = val_generator.classes

# Classification Report
print("\n" + "-"*40)
print("CLASSIFICATION REPORT")
print("-"*40)
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

 
axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix', fontsize=14)
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

if 'lr' in history.history:
    axes[1, 1].plot(history.history['lr'], linewidth=2, color='green')
    axes[1, 1].set_title('Learning Rate', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].grid(True, alpha=0.3)
else:
    result_text = f"""
    MODEL RESULTS
    ================
    
    Final Accuracy: {val_accuracy*100:.2f}%
    Final Loss: {val_loss:.4f}
    
    Model: MobileNetV2
    Input Size: {IMG_SIZE}x{IMG_SIZE}
    Batch Size: {BATCH_SIZE}
    
    Total Education: {len(history.history['accuracy'])} epoch
    """
    axes[1, 1].text(0.1, 0.5, result_text, fontsize=12, 
                    verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
plt.show()


model.save('mask_detection_mobilenetv2_final.h5')
print("Model Saved.")
# Finished.