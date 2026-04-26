import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('best_custom_cnn.keras')

def predict_teeth(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    
    if prediction[0][0] > 0.5:
        result = f"Healthy (Confidence: {confidence*100:.2f}%)"
    else:
        result = f"Unhealthy (Confidence: {confidence*100:.2f}%)"
    
    return result

# Test
if __name__ == "__main__":
    result = predict_teeth("15.jpg")
    result2 = predict_teeth("25.jpg")
    result3 = predict_teeth("100.jpg")
    result4 = predict_teeth("30.jpg")
    print(f"The teeth image is classified as: {result}")
    print(f"The teeth image is classified as: {result2}")
    print(f"The teeth image is classified as: {result3}")
    print(f"The teeth image is classified as: {result4}")

