# Import necessary libraries;
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import warnings
warnings.filterwarnings("ignore")

# Load dataset, train/test split, normalization and preprocessing for teeth images;
healthy_path = "healthy"
unhealthy_path = "unhealthy"

healthy_files = pd.DataFrame({'filename':os.listdir(healthy_path)})
healthy_files['path'] = healthy_files['filename'].apply(lambda x: os.path.join(healthy_path, x)).astype(str)
healthy_files['label'] = "1"  

unhealthy_files = pd.DataFrame({'filename':os.listdir(unhealthy_path)})
unhealthy_files['path'] = unhealthy_files['filename'].apply(lambda x: os.path.join(unhealthy_path, x)).astype(str)
unhealthy_files['label'] = "0"

df = pd.concat([healthy_files, unhealthy_files], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Data augmentation;
datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,  
    x_col='path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Build the model using EfficientNetB0 as base;
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True  # Freeze the base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model;
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)],
    verbose = 1
)

# Valuate the model;
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix;

Y_pred = model.predict(test_generator)
y_pred = (Y_pred > 0.5).astype(int)

y_true = test_generator.labels
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unhealthy', 'Healthy'], yticklabels=['Unhealthy', 'Healthy'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=['Unhealthy', 'Healthy']))

# Testing;
img_path = "10.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

img_array = preprocess_input(tf.keras.preprocessing.image.img_to_array(img))
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("The teeth image is classified as: Healthy")
else:
    print("The teeth image is classified as: Unhealthy")