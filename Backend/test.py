import cv2
import numpy as np
import json
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing import image
from PIL import Image

# ---------- Load Snake Detector ----------
snake_detector = load_model("snake_detector.keras")

def predict_snake(img_path):
    """Check if image contains a snake (binary classifier)."""
    # Use PIL to match Colab preprocessing
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = snake_detector.predict(img_array)[0][0]
    if pred > 0.5:
        return True, "Snake Detected 🐍"
    else:
        return False, "No Snake 🚫"

# ---------- Species Predictor ----------
def predictor(img_path, lang="en"):
    # --- Step 0: Snake validation ---
    is_snake, message = predict_snake(img_path)
    if not is_snake:
        return {"warning": f"⚠️ {message}. Please upload a snake image."}
    print(message)

    # --- Define and register focal loss (for loading your main model) ---
    @register_keras_serializable(package="Custom", name="focal_loss_fixed")
    def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=0.25):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = K.cast(y_true, K.floatx())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)

    @register_keras_serializable(package="Custom", name="focal_loss")
    def focal_loss(gamma=2., alpha=0.25):
        return lambda y_true, y_pred: focal_loss_fixed(y_true, y_pred, gamma, alpha)

    custom_objects = {
        "focal_loss_fixed": focal_loss_fixed,
        "focal_loss": focal_loss
    }

    # --- Load species classification model ---
    model = load_model("res_final_finetuned_model.keras", custom_objects=custom_objects)

    # --- Load class indices ---
    with open("class_indices.json", "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    # --- Load multilingual snake data ---
    with open("model_content_ml.json", "r", encoding="utf-8") as f:
        snake_data = json.load(f)

    # --- Preprocess image consistently using PIL ---
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    image_array = np.array(img, dtype="float32")
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    # --- Predict species ---
    predictions = model.predict(image_array)
    predicted_class_index = int(predictions.argmax())

    dict_class = {v: k for k, v in class_indices.items()}
    predicted_class_name = dict_class.get(predicted_class_index, "Unknown")

    # --- Fetch details in requested language ---
    details = {}
    if predicted_class_name in snake_data:
        for field, translations in snake_data[predicted_class_name].items():
            details[field] = translations.get(lang, translations.get("en", "N/A"))

    return details

# ---------------- Example ----------------
if __name__ == "__main__":
    test_img = "/content/WhatsApp Image 2025-09-13 at 14.41.11_332d447c.jpg"
    result = predictor(test_img, lang="ta")  # Tamil example
    print(result)
