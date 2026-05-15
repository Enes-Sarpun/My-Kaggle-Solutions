# Import necessary libraries;
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
from tensorflow.keras.optimizers import Adam, AdamW
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

# Load dataset, train/test split, normalization, Data Augmentation and preprocessing for brain images;

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
base_path = os.path.join(parent_dir, 'brain_tumor_dataset')

class_names = ['glioma', 'meningioma', 'healthy', 'pituitary']

for class_name in class_names:
    class_path = os.path.join(base_path, class_name)
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Directory {class_path} does not exist.")
    num_images = len(os.listdir(class_path))
    print(f"{class_name:15}: {num_images} images.")

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

# We made CNN Network like a ResNet!

def squeeze_excitation_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]

    # Global Average Pooling;
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)

    # Two FC Layers;
    se = Dense(channels//ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(channels, activation='sigmoid', kernel_initializer='he_normal')(se)

    # Scale;
    return Multiply()([input_tensor, se])

def residual_block(x, filters, strides=1):
    shortcut = x

    # First Conv2D;
    x = Conv2D(filters, (3,3), padding='same', kernel_initializer='he_normal', strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second Conv2D;
    x = Conv2D(filters, (3,3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # SEB;
    x = squeeze_excitation_block(x)

    # Shortcut Connection;
    if strides!=1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters,(1,1),strides=strides,padding='same',kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x,shortcut])
    x = Activation('relu')(x)
    
    return x

def conv_blocks(x, filters, kernel_size=3, strides=1):
    x = Conv2D(filters, (kernel_size, kernel_size), strides=strides, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def Build_CNN_Networks(input_shape=(224,224,3)):
    inputs = Input(shape=input_shape)

    # Firstly;
    x = conv_blocks(inputs,32,kernel_size=3,strides=2)
    x = conv_blocks(x,64,kernel_size=3,strides=1)
    x = MaxPooling2D((2,2))(x)

    # Stage 1: 64 Filters;
    x = residual_block(x,64)
    x = residual_block(x,64)

    # Stage 2: 128 Filters;
    x = residual_block(x,128,strides=2)
    x = residual_block(x,128)
    x = residual_block(x,128)

    # Stage 3: 256 Filters;
    x = residual_block(x,256,strides=2)
    x = residual_block(x,256)
    x = residual_block(x,256)

    # Stage 4: 512 Filters;
    x = residual_block(x,512,strides=2)
    x = residual_block(x,512)

    # Global Average Pooling;
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    x = Concatenate()([avg_pool, max_pool])

    # Classifier Head;
    x = Dense(512,kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(256,kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(4, activation='softmax')(x)  # 4 sınıf için softmax
    model = Model(inputs, outputs, name='Custom_Brain_CNN')

    return model

# Data Augmentation and Preprocessing for dataset;

data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_set = data_gen.flow_from_directory(
    base_path,  
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

test_set = test_gen.flow_from_directory(
    base_path,  
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Model Compilation, Training and Evaluation will be here...

model = Build_CNN_Networks(input_shape=(224,224,3))
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),  
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

history = model.fit(
    train_set,
    epochs=10,
    validation_data=test_set,
    callbacks=callbacks
)

# Evaluate the model;
loss, accuracy = model.evaluate(test_set)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




# ACC = %92