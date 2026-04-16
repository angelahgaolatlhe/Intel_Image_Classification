import os
import io
import numpy as np

from flask import Flask, request, jsonify, render_template
from PIL import Image

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 150
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

PYTORCH_MODEL_PATH    = os.path.join(os.path.dirname(__file__), "angelah_model.pth")
TENSORFLOW_MODEL_PATH = os.path.join(os.path.dirname(__file__), "angelah_model.keras")

# ── Lazy model cache ───────────────────────────────────────────────────────────
_pytorch_model = None
_tf_model      = None
_device        = None


def get_pytorch_model():
    global _pytorch_model, _device
    if _pytorch_model is None:
        import torch
        import torch.nn as nn
        from torchvision import transforms

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class CNNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 256), nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 6)
                )
            def forward(self, x):
                return self.fc(self.conv(x))

        print("[INFO] Loading PyTorch model...")
        model = CNNModel().to(_device)
        model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=_device))
        model.eval()
        _pytorch_model = (model, transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
        print("[INFO] PyTorch model ready.")
    return _pytorch_model


def get_tf_model():
    global _tf_model
    if _tf_model is None:
        import tensorflow as tf
        print("[INFO] Loading TensorFlow model...")
        _tf_model = tf.keras.models.load_model(TENSORFLOW_MODEL_PATH)
        print("[INFO] TensorFlow model ready.")
    return _tf_model


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_for_pytorch(image):
    import torch
    _, transform = get_pytorch_model()
    image = image.convert("RGB")
    return transform(image).unsqueeze(0).to(_device)


def preprocess_for_tensorflow(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.array(image, dtype=np.float32), axis=0)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file      = request.files["image"]
    framework = request.form.get("framework", "pytorch").lower()

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if framework not in ("pytorch", "tensorflow"):
        return jsonify({"error": "Invalid framework."}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    try:
        if framework == "pytorch":
            import torch
            model, _ = get_pytorch_model()
            tensor = preprocess_for_pytorch(image)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_idx].item() * 100
        else:
            arr = preprocess_for_tensorflow(image)
            predictions = get_tf_model().predict(arr, verbose=0)
            predicted_idx = int(np.argmax(predictions, axis=1)[0])
            confidence = float(predictions[0][predicted_idx]) * 100

        return jsonify({
            "class":      CLASS_NAMES[predicted_idx],
            "confidence": f"{confidence:.1f}%",
            "framework":  framework
        })

    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
