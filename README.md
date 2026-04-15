# SceneLens — Intel Image Classification

A dual-framework image classification project that trains and deploys two
convolutional neural networks in **PyTorch** and in **TensorFlow**
to classify natural and urban scenes from the
[Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

The project includes a **Flask web application** called SceneLens where users can upload a photo
and get a predicted scene class from either model.

---

## Recognised Scene Classes

| Class      | Description                        |
|------------|------------------------------------|
| Buildings  | Urban buildings and architecture   |
| Forest     | Dense trees and woodland           |
| Glacier    | Ice fields and glacial landscapes  |
| Mountain   | Mountain peaks and rocky terrain   |
| Sea        | Ocean, sea, and coastal scenes     |
| Street     | Roads, pavements, and streetscapes |

---

## Project Structure

```
Intel_Image_Classification/
├── app.py                    # Flask backend — loads both models and serves /predict
├── main.py                   # CLI training entry point (--framework pytorch|tensorflow)
├── pytorch_model.py          # PyTorch CNN definition and training loop
├── tensorflow_model.py       # TensorFlow CNN definition and training loop
├── pytorch_evaluation.py     # PyTorch evaluation script
├── tensorflow_evaluation.py  # TensorFlow evaluation script
├── angelah_model.pth         # Saved PyTorch model weights (generated after training and you can use your firstname instead)
├── angelah_model.keras       # Saved TensorFlow model (generated after training and you can use your firstname instead)
├── templates/
│   └── index.html            # Two-screen web interface (welcome + classifier)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Dependencies

Python **3.9 or higher** is recommended.

| Package        | Purpose                                  |
|----------------|------------------------------------------|
| flask          | Web server and routing                   |
| torch          | PyTorch model training and inference     |
| torchvision    | Image transforms and dataset utilities  |
| tensorflow     | TensorFlow model training and inference  |
| Pillow         | Image loading and preprocessing in Flask |
| scikit-learn   | Classification report and confusion matrix |
| numpy          | Array operations                         |
| matplotlib     | Plotting (evaluation notebooks)          |
| seaborn        | Confusion matrix heatmaps                |

---

## Installation

### 1. Clone or unzip the project

```bash
cd Intel_classification_models
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** The default `torch` and `tensorflow` packages above use CPU.
> If your machine has an NVIDIA GPU, install the GPU-enabled versions instead:
>
> ```bash
> # PyTorch with CUDA 11.8
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
>
> # TensorFlow GPU support is included automatically from TF 2.x onwards
> # as long as CUDA and cuDNN are correctly installed on your system.
> ```

---

## Dataset Setup

1. Download the dataset from Kaggle:
   [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

2. Extract and place it in a `data/` folder inside the project root:

```
Intel_Image_Classification/
└── data/
    ├── seg_train/
    │   ├── buildings/
    │   ├── forest/
    │   ├── glacier/
    │   ├── mountain/
    │   ├── sea/
    │   └── street/
    └── seg_test/
        ├── buildings/
        └── ...
```

---

## Training

Use `main.py` to train either model from the command line.

**Train the PyTorch model:**
```bash
python main.py --framework pytorch
```

**Train the TensorFlow model:**
```bash
python main.py --framework tensorflow
```

Each training run saves the model to the project root:
- PyTorch  → `angelah_model.pth`
- TensorFlow → `angelah_model.keras`

Both models train for **10 epochs** at `150×150` input resolution with a
batch size of 32. GPU is used automatically if available.

---

## Evaluation

**PyTorch evaluation:**
```bash
python pytorch_evaluation.py
# Outputs: accuracy, per-class report, confusion matrix saved to working directory
```

**TensorFlow evaluation:**
```bash
python tensorflow_evaluation.py
# Outputs: loss, accuracy, per-class report, confusion matrix saved to working directory
```

---

## Running the Web Application

Make sure both model files (`angelah_model.pth` and `angelah_model.keras`)
are present in the project root before starting the server.

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

The web app has two screens:

1. **Welcome screen** — describes the app and how to use it, with a
   "Get Started" button.
2. **Classifier screen** — select a model (PyTorch or TensorFlow), upload a
   JPG / PNG / WEBP image, click "Classify Image", and see the predicted
   scene class with a confidence score.

---

## Deployment on PythonAnywhere

1. Create a free account at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload the project files via the **Files** tab or `git clone`
3. Upload `angelah_model.pth` and `angelah_model.keras` to the project root
4. Open a **Bash console** and install dependencies:
   ```bash
   pip install --user -r requirements.txt
   ```
5. Go to the **Web** tab → **Add a new web app** → choose **Flask**
6. Set the **Source code** directory to your project folder
7. Set the **WSGI file** to point to `app.py` and ensure the Flask `app`
   object is correctly referenced
8. Click **Reload** and visit your assigned `.pythonanywhere.com` URL

---

# Author

Angelah Kgato Gaolatlhe

---

# License

This project is for academic purposes.
