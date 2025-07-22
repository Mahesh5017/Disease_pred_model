import os
import uuid
import sys
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Ensure scripts folder is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.inference import predict_image  # Make sure this file exists

app = Flask(__name__)
CORS(app)  # Allow requests from any origin

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    print("📥 Received a prediction request.")

    # Check if image is present in the request
    if 'image' not in request.files:
        print("❌ No image file found in the request.")
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        print("❌ No file selected.")
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    print(f"📸 Image saved at: {filepath}")

    try:
        # Call the model inference
        result = predict_image(filepath)
        print(f"✅ Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    finally:
        # Always attempt to delete the file
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🧹 Deleted uploaded file: {filepath}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
