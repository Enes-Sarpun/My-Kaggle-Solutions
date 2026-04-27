from pathlib import Path
import shutil
import random
import PIL  
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array, image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight


class PretrainedModel:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.class_names = ["ai", "real"]
        self.class_weights = None
        self.project_root = Path(__file__).resolve().parent.parent
        self.best_model_path = self.project_root / "best_model.keras"
        self.base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape
        )
        self.model = self.build_model()

    def _resolve_path(self, path_value):
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.project_root / path

    def prepare_data(self, source_path="datas", target_path="data_split", val_ratio=0.2, seed=42):
        target = self._resolve_path(target_path)
        if target.exists() and any(target.iterdir()):
            print(f"'{target}' skipping.")
            return

        random.seed(seed)
        source = self._resolve_path(source_path)

        for split in ["train", "val"]:
            for cls in ["ai", "real"]:
                (target / split / cls).mkdir(parents=True, exist_ok=True)

        mapping = {
            "Ai_generated_dataset": "ai",
            "real_dataset": "real"
        }

        for src_folder, cls_name in mapping.items():
            all_images = []
            for category in (source / src_folder).iterdir():
                if category.is_dir():
                    all_images.extend(list(category.glob("*.*")))
            random.shuffle(all_images)
            val_count = int(len(all_images) * val_ratio)
            val_images = all_images[:val_count]
            train_images = all_images[val_count:]

            for img in train_images:
                shutil.copy(img, target / "train" / cls_name / img.name)
            for img in val_images:
                shutil.copy(img, target / "val" / cls_name / img.name)

            print(f"'{cls_name}': {len(train_images)} train, {len(val_images)} val")

    def compute_class_weights(self, target_path="data_split"):
        train_path = self._resolve_path(target_path) / "train"
        labels = []
        for idx, cls in enumerate(self.class_names):
            count = len(list((train_path / cls).glob("*.*")))
            labels.extend([idx] * count)
            print(f"'{cls}': {count} images.")

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

    def augment_data(self):
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
            layers.RandomZoom(0.1),
            layers.RandomRotation(0.05), 
        ])

    def build_model(self):
        self.base_model.trainable = False

        inputs = layers.Input(shape=self.input_shape)
        x = self.augment_data()(inputs)
        x = preprocess_input(x)
        x = self.base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss=losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"]
        )
        return model

    def train(self, train_ds, val_ds, epochs_stage1=15, epochs_stage2=45):
        # Stage 1: Only Classifier
        cb1 = [
            callbacks.ModelCheckpoint(str(self.best_model_path), save_best_only=True, monitor="val_accuracy"),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=2, monitor="val_loss")
        ]
        print("\n--- Stage 1: Classifier training ---")
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage1,
            callbacks=cb1,
            class_weight=self.class_weights
        )

        # Stage 2: All Model, Fine-tune
        print("\n--- Stage 2: Fine-tune ---")
        self.base_model.trainable = True
        self.model.compile(
            optimizer=optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
            loss=losses.BinaryCrossentropy(label_smoothing=0.05),
            metrics=["accuracy"]
        )
        cb2 = [
            callbacks.ModelCheckpoint(str(self.best_model_path), save_best_only=True, monitor="val_accuracy"),
            callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=2, monitor="val_loss")
        ]
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage2,
            callbacks=cb2,
            class_weight=self.class_weights
        )

        best_val_acc = max(history.history["val_accuracy"])
        print(f"The best validation accuracy: {best_val_acc:.4f}")
        return history


class Test:
    def __init__(self, class_names, model_path="best_model.keras"):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names

    def predict(self, img_path):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        pred = self.model.predict(img_array, verbose=0)[0][0]
        label = self.class_names[int(pred > 0.5)]
        confidence = pred if pred > 0.5 else 1 - pred
        return label, float(confidence)

    def predict_with_tta(self, img_path):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)

        normal = np.expand_dims(img_array, axis=0)
        flipped = np.expand_dims(np.fliplr(img_array), axis=0)

        pred1 = self.model.predict(normal, verbose=0)[0][0]
        pred2 = self.model.predict(flipped, verbose=0)[0][0]
        avg_pred = (pred1 + pred2) / 2

        label = self.class_names[int(avg_pred > 0.5)]
        confidence = avg_pred if avg_pred > 0.5 else 1 - avg_pred
        return label, float(confidence)

if __name__ == "__main__":
    pm = PretrainedModel()
    pm.prepare_data()
    pm.compute_class_weights()       
    train_ds, val_ds = pm.load_datasets()
    pm.train(train_ds, val_ds)

    # Test
    test_path = pm.project_root / "test_image.png"
    if test_path.exists():
        tester = Test(class_names=pm.class_names, model_path=str(pm.best_model_path))
        label, conf = tester.predict(str(test_path))
        print(f"Predict: {label} ({conf*100:.2f}%)")

        label2, conf2 = tester.predict_with_tta(str(test_path))
        print(f"Predict (TTA): {label2} ({conf2*100:.2f}%)")
    else:
        print(f"'{test_path}' not found, test skipped.")

# Finished.