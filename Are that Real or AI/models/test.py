import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

# FFTLayer'ı import et (custom_objects için lazım)
sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from frequency_model import FFTLayer

project_root = Path.cwd()
model = tf.keras.models.load_model(
    project_root / "best_models" / "best_hybrid_model.keras",
    custom_objects={"FFTLayer": FFTLayer}
)

# Test görselini yükle (Gemini'den ürettiğin kaplan)
img = Image.open(project_root / "test_image_real.jpg").convert("RGB").resize((224, 224))
img_array = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

pred = model.predict(img_array, verbose=0)[0][0]
class_names = ["ai", "real"]  # alfabetik
label = class_names[int(pred > 0.5)]
confidence = pred if pred > 0.5 else 1 - pred

print(f"Raw output: {pred:.4f}")
print(f"Prediction: {label}")
print(f"Confidence: {confidence*100:.2f}%")