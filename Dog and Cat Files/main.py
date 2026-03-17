# Import Necessary Libraries;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import os

# Data Loading;
data_path = "PetImages"
IMG_SIZE = (224, 224)
Batch_Size = 32

# Training Dataset;
train_ds = image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=Batch_Size,
    label_mode='binary'
)

# Validation Dataset;
val_ds = image_dataset_from_directory(
    data_path, 
    validation_split=0.2,
    subset='validation',
    seed=123,  
    image_size=IMG_SIZE,
    batch_size=Batch_Size,
    label_mode='binary'
)

class_names = train_ds.class_names
print("Class names: ",class_names)
print("--"*25)

# Visualizing some images from the training dataset;
for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
    plt.show()

# Data Preprocessing Function;
train_ds = train_ds.map(lambda x,y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x,y: (preprocess_input(x), y))

# Performance Optimization;
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Model;
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freezing the base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.summary()

# Model Optimization;
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks, Early Stopping, Checkpointing;
Early_Stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
Model_Checkpoint = ModelCheckpoint('best_model_mobilnet.h5', monitor='val_loss', save_best_only=True)

# Model Training;
Epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=Epochs,
    callbacks=[Early_Stopping, Model_Checkpoint]
)

# Historical data;
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))
plt.figure(figsize=(12, 6))

# Accuracy Plots;
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Başarısı (Train Acc)')
plt.plot(epochs_range, val_acc, label='Doğrulama Başarısı (Val Acc)')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Başarısı')

# Loss Plots;
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Hatası (Train Loss)')
plt.plot(epochs_range, val_loss, label='Doğrulama Hatası (Val Loss)')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Hatası')
plt.show()

print("Training Complete.")
