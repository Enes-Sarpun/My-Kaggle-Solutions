import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # veya Path.cwd()
model_path = project_root / "best_models" / "best_spatial_model.keras"
test_image_path = project_root / "test_image_gemini.png"  # dosya adına göre güncelle

model = tf.keras.models.load_model(model_path)
class_names = ["ai", "real"]

img = Image.open(test_image_path).convert("RGB").resize((224, 224))
img_array = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

pred = model.predict(img_array, verbose=0)[0][0]
label = class_names[int(pred > 0.5)]
confidence = pred if pred > 0.5 else 1 - pred

print(f"Raw output: {pred:.4f}")
print(f"Prediction: {label} ({confidence*100:.2f}%)")