from flask import Flask, request, jsonify
from test import predictor
import tempfile
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Snake Classifier API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # ✅ Get language from request (default English if not provided)
    lang = request.form.get("language", "en")

    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        temp_path = tmp.name
        file.save(temp_path)

    try:
        # ✅ Pass language into predictor
        result = predictor(temp_path, lang=lang)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
