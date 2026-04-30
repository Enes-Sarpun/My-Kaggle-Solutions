from pathlib import Path
import numpy as np
import tensorflow as tf
from frequency_model import FFTLayer
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight


class HybridModel:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.class_names = ["ai", "real"]
        self.class_weights = None
        self.project_root = Path(__file__).resolve().parent.parent
        self.best_model_path = self.project_root / "best_models" / "best_hybrid_model.keras"
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        self.spatial_model_path = self.project_root / "best_models" / "best_spatial_model.keras"
        self.frequency_model_path = self.project_root / "best_models" / "best_frequency_model.keras"

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

    def _load_pretrained(self, path, name_prefix):
        full_model = tf.keras.models.load_model(str(path), custom_objects={"FFTLayer": FFTLayer} )

        feature_layer = full_model.layers[-3].output  
        
        feature_extractor = models.Model(
            inputs=full_model.input,
            outputs=feature_layer,
            name=name_prefix
        )
        
        feature_extractor.trainable = False
        for layer in feature_extractor.layers:
            layer.trainable = False
        
        return feature_extractor

    def build_model(self):
        print("Loading spatial model...")
        spatial_extractor = self._load_pretrained(
            self.spatial_model_path, "spatial_features"
        )
        print(f"Spatial output shape: {spatial_extractor.output_shape}")

        print("Loading frequency model...")
        frequency_extractor = self._load_pretrained(
            self.frequency_model_path, "frequency_features"
        )
        print(f"Frequency output shape: {frequency_extractor.output_shape}")

        inputs = layers.Input(shape=self.input_shape, name="image_input")

        spatial_feat = spatial_extractor(inputs, training=False)
        frequency_feat = frequency_extractor(inputs, training=False)

        # Feature concatenation
        merged = layers.Concatenate(name="fusion")([spatial_feat, frequency_feat])

        # Fusion classifier
        x = layers.Dense(256, activation="relu")(merged)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = models.Model(inputs, outputs, name="hybrid_model")
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=losses.BinaryCrossentropy(label_smoothing=0.05),
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
                patience=6,
                restore_best_weights=True,
                monitor="val_accuracy"
            ),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss")
        ]

        print("\n--- Hybrid Model Training ---")
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
    hm = HybridModel()
    hm.model.summary()

    hm.compute_class_weights()
    train_ds, val_ds = hm.load_datasets()
    hm.train(train_ds, val_ds, epochs=20)




# Finished.