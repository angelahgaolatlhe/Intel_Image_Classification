"""
Flask web application for Intel Image Classification.
Serves both PyTorch (.pth) and TensorFlow (.keras) models.
"""

import os
import io
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─────────────────────────────────────────────
# Class labels and metadata
# ─────────────────────────────────────────────
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
CLASS_EMOJIS = {
    'buildings': '🏙️',
    'forest':    '🌲',
    'glacier':   '🧊',
    'mountain':  '⛰️',
    'sea':       '🌊',
    'street':    '🛣️',
}
CLASS_DESCRIPTIONS = {
    'buildings': 'Urban architecture and city structures',
    'forest':    'Dense woodland and natural forest areas',
    'glacier':   'Ice formations and frozen landscapes',
    'mountain':  'High-altitude rocky terrain and peaks',
    'sea':       'Ocean, coastline and marine environments',
    'street':    'Roads, paths and urban streetscapes',
}
IMG_SIZE = 150

# ─────────────────────────────────────────────
# Lazy model loading
# ─────────────────────────────────────────────
_pytorch_model = None
_tf_model = None
_device = None


def load_pytorch_model():
    global _pytorch_model, _device
    if _pytorch_model is not None:
        return _pytorch_model

    import torch
    from train_pytorch import IntelCNN

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelCNN(num_classes=len(CLASSES)).to(_device)

    pth_path = os.path.join(os.path.dirname(__file__), "student_model.pth")
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"PyTorch model not found at: {pth_path}")

    checkpoint = torch.load(pth_path, map_location=_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    _pytorch_model = model
    print(f"[PyTorch] Model loaded on {_device}")
    return model


def load_tf_model():
    global _tf_model
    if _tf_model is not None:
        return _tf_model

    import tensorflow as tf
    keras_path = os.path.join(os.path.dirname(__file__), "student_model.keras")
    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"TensorFlow model not found at: {keras_path}")

    _tf_model = tf.keras.models.load_model(keras_path)
    print("[TensorFlow] Model loaded")
    return _tf_model


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
def preprocess_pytorch(img: Image.Image):
    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(img).unsqueeze(0)   # (1, 3, H, W)
    return tensor


def preprocess_tf(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)   # (1, H, W, 3)


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict_pytorch(img: Image.Image):
    import torch
    model = load_pytorch_model()
    tensor = preprocess_pytorch(img).to(_device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def predict_tf(img: Image.Image):
    model = load_tf_model()
    arr = preprocess_tf(img)
    probs = model.predict(arr, verbose=0)[0]
    return probs


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           classes=CLASSES,
                           class_emojis=CLASS_EMOJIS)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    model_choice = request.form.get('model', 'pytorch')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate extension
    allowed = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Unsupported file type: .{ext}'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        if model_choice == 'pytorch':
            probs = predict_pytorch(img)
            framework = 'PyTorch'
        else:
            probs = predict_tf(img)
            framework = 'TensorFlow'

        top_idx = int(np.argmax(probs))
        top_class = CLASSES[top_idx]
        top_conf = float(probs[top_idx])

        # Top-3 predictions
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                'label': CLASSES[i],
                'emoji': CLASS_EMOJIS[CLASSES[i]],
                'probability': float(probs[i]),
                'description': CLASS_DESCRIPTIONS[CLASSES[i]],
            }
            for i in top3_idx
        ]

        return jsonify({
            'success': True,
            'predicted_class': top_class,
            'confidence': top_conf,
            'emoji': CLASS_EMOJIS[top_class],
            'description': CLASS_DESCRIPTIONS[top_class],
            'framework': framework,
            'top3': top3,
            'all_probs': {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        })

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'classes': CLASSES})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
