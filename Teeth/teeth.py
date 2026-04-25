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

# Custom CNN model blocks;

def squeeze_excitation_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation block - channel attention mechanism"""
    channels = input_tensor.shape[-1]
    
    # Squeeze: Global Average Pooling
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    
    # Excitation: Two FC layers
    se = Dense(channels // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(channels, activation='sigmoid', kernel_initializer='he_normal')(se)
    
    # Scale
    return Multiply()([input_tensor, se])

def residual_block(x, filters, strides=1):
    """Residual Block with SE attention"""
    shortcut = x
    
    # First Conv
    x = Conv2D(filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second Conv
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    
    # SE Block
    x = squeeze_excitation_block(x)
    
    # Shortcut connection
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def conv_block(x, filters, kernel_size=3, strides=1):
    """Standard Conv Block with BN and ReLU"""
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_custom_cnn(input_shape=(224, 224, 3)):
    """
    Custom Deep CNN with:
    - Residual Connections (Skip Connections)
    - Squeeze-and-Excitation Attention
    - Progressive feature extraction
    """
    inputs = Input(shape=input_shape)
    
    # STEM: İlk özellik çıkarma
    x = conv_block(inputs, 32, kernel_size=3, strides=2)  # 112x112
    x = conv_block(x, 64, kernel_size=3, strides=1)
    x = MaxPooling2D((2, 2))(x)  # 56x56
    
    # Stage 1: 64 filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    # Stage 2: 128 filters
    x = residual_block(x, 128, strides=2)  # 28x28
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    # Stage 3: 256 filters
    x = residual_block(x, 256, strides=2)  # 14x14
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    # Stage 4: 512 filters
    x = residual_block(x, 512, strides=2)  # 7x7
    x = residual_block(x, 512)
    
    # Global Pooling - hem Average hem Max kullan
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    # Classifier Head
    x = Dense(512, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='Custom_ResNet_SE')
    return model


# MAIN TRAINING CODE - Runs only when executed directly;

if __name__ == "__main__":
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

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])

    # Class weights;
    class_weights = compute_class_weight('balanced', classes=np.array(['0', '1']), y=train_df['label'].values)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class Weights: {class_weight_dict}")

    # Data Augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=None,
        x_col='path',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=None, 
        x_col='path',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    # Create a Model;

    model = build_custom_cnn(input_shape=(224, 224, 3))

    # Cosine Annealing Learning Rate Schedule;
    def cosine_decay_with_warmup(epoch, lr):
        warmup_epochs = 5
        total_epochs = 50
        initial_lr = 0.001
        min_lr = 1e-6
        
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    print(f"\nTotal number of parameters: {model.count_params():,}")

    # Model Training;

    print("=" * 60)
    print("TRAINING CUSTOM CNN WITH RESIDUAL + SE BLOCKS")
    print("=" * 60)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        ModelCheckpoint('best_custom_cnn.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        LearningRateScheduler(cosine_decay_with_warmup, verbose=0)
    ]

    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=test_generator,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Load the best model;
    model = tf.keras.models.load_model('best_custom_cnn.keras', custom_objects={
        'squeeze_excitation_block': squeeze_excitation_block,
        'residual_block': residual_block,
        'conv_block': conv_block
    })

    # Valuate the model;
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
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

    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    if prediction[0][0] > 0.5:
        print(f"The teeth image is classified as: Healthy (Confidence: {confidence*100:.2f}%)")
    else:
        print(f"The teeth image is classified as: Unhealthy (Confidence: {confidence*100:.2f}%)")
