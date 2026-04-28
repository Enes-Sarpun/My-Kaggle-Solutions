from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight


class FFTLayer(layers.Layer):

    def call(self, inputs):
        # RGB -> Grayscale
        gray = tf.image.rgb_to_grayscale(inputs)
        gray = tf.squeeze(gray, axis=-1)

        # 2D FFT
        gray_complex = tf.cast(gray, tf.complex64)
        fft = tf.signal.fft2d(gray_complex)
        fft_shifted = tf.signal.fftshift(fft, axes=[-2, -1])

        # Log magnitude
        magnitude = tf.abs(fft_shifted)
        log_magnitude = tf.math.log(magnitude + 1.0)

        # Per-image normalization
        mean = tf.reduce_mean(log_magnitude, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(log_magnitude, axis=[1, 2], keepdims=True)
        normalized = (log_magnitude - mean) / (std + 1e-6)

        return tf.expand_dims(normalized, axis=-1)


class FrequencyModel:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.class_names = ["ai", "real"]
        self.class_weights = None
        self.project_root = Path(__file__).resolve().parent.parent
        self.best_model_path = self.project_root / "best_models" / "best_frequency_model.keras"
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model = self.build_model()

    def _resolve_path(self, path_value):
        path = Path(path_value)
        return path if path.is_absolute() else self.project_root / path

    def compute_class_weights(self, target_path="data_split"):
        train_path = self._resolve_path(target_path) / "train"
        labels = []
        for idx, cls in enumerate(self.class_names):
            count = len(list((train_path / cls).glob("*.*")))
            labels.extend([idx] * count)
            print(f"'{cls}': {count} images")

        labels = np.array(labels)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )
        self.class_weights = {i: float(w) for i, w in enumerate(weights)}
        print(f"Class weights: {self.class_weights}")
        return self.class_weights

    def load_datasets(self, target_path="data_split", batch_size=32):
        base_path = self._resolve_path(target_path)
        
        train_ds = image_dataset_from_directory(
            str(base_path / "train"),
            image_size=self.input_shape[:2],
            batch_size=batch_size,
            label_mode="binary",
            shuffle=True
        )
        val_ds = image_dataset_from_directory(
            str(base_path / "val"),
            image_size=self.input_shape[:2],
            batch_size=batch_size,
            label_mode="binary",
            shuffle=False
        )

        self.class_names = train_ds.class_names
        print("Classes:", self.class_names)

        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)

        return train_ds, val_ds

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        
        x = FFTLayer(name="fft_transform")(inputs)
        
        # CNN Block 1
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # CNN Block 2
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # CNN Block 3
        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = models.Model(inputs, outputs, name="frequency_stream")
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"]
        )
        return model

    def train(self, train_ds, val_ds, epochs=20):
        cb = [
            callbacks.ModelCheckpoint(
                str(self.best_model_path),
                save_best_only=True,
                monitor="val_accuracy"
            ),
            callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor="val_accuracy"
            ),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=2, monitor="val_loss")
        ]

        print("\n--- Frequency Stream Training ---")
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=cb,
            class_weight=self.class_weights
        )

        best_val_acc = max(history.history["val_accuracy"])
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")
        return history


if __name__ == "__main__":
    fm = FrequencyModel()
    fm.model.summary()
    
    fm.compute_class_weights()
    train_ds, val_ds = fm.load_datasets()
    fm.train(train_ds, val_ds, epochs=20)


# Finished.