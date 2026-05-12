# Import Necessary Libraries;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# GPU Configuration + Mixed Precision (2x faster);
gpus = tf.config.experimental.list_physical_devices('GPU 0')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Mixed precision for faster training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled!")
    except RuntimeError as e:
        print(e)

# Load Dataset, Preprocessing, and Augmentation;

train_datagen = ImageDataGenerator( # Data Augmentation;
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
)

IMG_SIZE = 224 # Image and Batch Size;
BATCH_SIZE = 64

train_generator = train_datagen.flow_from_directory( # Train and Validation will be created from same generator;
    'data',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Classes: {train_generator.class_indices}")
print(f"Images Shape", train_generator.image_shape)

# Train-Test Split and Model Definition;

model = Sequential([ # Model Definition;
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),  
    Dropout(0.5),
    Dense(1, activation='sigmoid', dtype='float32')
])

model.summary() # Model Summary; (to check parameters and layers .etc);
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile the Model so it is ready for training;

# Model Compilation, Training, and Evaluation;
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Early Stopping to prevent overfitting;
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True) # Save the best model;
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6) # Reduce LR on Plateau, It will be helpful for fine tuning;

history = model.fit( # Model Training;
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=5,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
)

# Plot Training History;
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue', alpha=0.7, linewidth=1,markersize=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', alpha=0.7, linewidth=1,markersize=2)
plt.xlabel('Epochs',fontsize=10,alpha=0.7)
plt.ylabel('Loss',fontsize=10,alpha=0.7)
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green', alpha=0.7, linewidth=1,markersize=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange',alpha=0.7, linewidth=1,markersize=2)
plt.xlabel('Epochs',fontsize=10,alpha=0.7)
plt.ylabel('Accuracy',fontsize=10,alpha=0.7)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Confusion Matrix and Classification Report;

validation_generator.reset()
Y_pred = model.predict(validation_generator, validation_generator.samples // validation_generator.batch_size + 1)
y_pred = np.round(Y_pred).astype(int).reshape(-1)
cm = confusion_matrix(validation_generator.classes, y_pred)
print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(classification_report(validation_generator.classes, y_pred, target_names=list(train_generator.class_indices.keys())))
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
class_labels = list(train_generator.class_indices.keys())
plot_confusion_matrix(cm, classes=class_labels, title='Confusion Matrix')


# Save the Model;   
model.save('mask_classification_model.h5')
print("Model saved as mask_classification_model.h5")


# Finished.