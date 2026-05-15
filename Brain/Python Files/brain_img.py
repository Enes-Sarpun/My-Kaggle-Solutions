# Import Necessary Libraries;
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                      BatchNormalization, GlobalAveragePooling2D,
                                      Add, Input, Activation, Multiply, Reshape,
                                      GlobalMaxPooling2D, Concatenate)
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import warnings
warnings.filterwarnings("ignore")


# GPU memory growth;
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# Define paths and class names
base_path = 'brain_tumor_dataset'
class_names = ['glioma', 'meningioma', 'healthy', 'pituitary']

# Count images in each class

for class_name in class_names:
    class_path = os.path.join(base_path, class_name)
    num_images = len(os.listdir(class_path))
    print(f"{class_name:15} : {num_images} images")

# Function to display sample images
def display_sample_images(base_path, class_names, samples_per_class=5):
    """Display sample images from each class in a grid"""
    
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(15, 12))
    fig.suptitle('Brain MRI Sample Images by Class', fontsize=16, fontweight='bold', y=1.02)
    
    for row, class_name in enumerate(class_names):
        class_path = os.path.join(base_path, class_name)
        image_files = os.listdir(class_path)[:samples_per_class]
        
        for col, img_file in enumerate(image_files):
            img_path = os.path.join(class_path, img_file)
            img = plt.imread(img_path)
            
            ax = axes[row, col]
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.axis('off')
            
            if col == 0:
                ax.set_ylabel(class_name.upper(), fontsize=12, fontweight='bold', rotation=0, 
                             labelpad=60, va='center')
    
    plt.tight_layout()
    plt.show()

# Check properties of first image from each class
for class_name in class_names:
    class_path = os.path.join(base_path, class_name)
    sample_img = os.listdir(class_path)[0]
    img_path = os.path.join(class_path, sample_img)
    img = plt.imread(img_path)
    
    print(f"\n{class_name.upper()}:")
    print(f"  - Image Shape: {img.shape}")
    print(f"  - Data Type: {img.dtype}")
    print(f"  - Min Value: {img.min()}")
    print(f"  - Max Value: {img.max()}")

# Train and Test Split, Data Generators and Class Weights;

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)    

train_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    base_path,
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical',
    subset='validation',
    shuffle=True  
)

# Display class indices
print("\nClass Indices:", train_generator.class_indices)

# Compute class weights to handle class imbalance;
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Callbacks;
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=10,
    restore_best_weights=True,
    verbose=1
)
model_checkpoint = ModelCheckpoint(
    'best_brain_tumor_model.h5',
    monitor='val_accuracy', 
    save_best_only=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5, 
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Model Training;
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,  
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# Plot Training History;
def plot_training_history(history):
    """Plot training & validation accuracy and loss values"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='orange')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='blue')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Evaluate Model on Validation Set;
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")


# Validation Loss: 0.0706
# Validation Accuracy: 0.9786