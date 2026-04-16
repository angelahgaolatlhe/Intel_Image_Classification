# Leitlho le le ntshotsho Intel Image Classification pipeline

A dual-framework image classification project that trains and deploys two
convolutional neural networks in **PyTorch** and in **TensorFlow**
to classify natural and urban scenes from the
[Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

The project includes a **Flask web application** called *Leitlho le le ntshotsho* where users can upload a photo
and get a predicted scene class from either model.

---

## Recognised Scene Classes

| Class     | Description                        |
| --------- | ---------------------------------- |
| Buildings | Urban buildings and architecture   |
| Forest    | Dense trees and woodland           |
| Glacier   | Ice fields and glacial landscapes  |
| Mountain  | Mountain peaks and rocky terrain   |
| Sea       | Ocean, sea, and coastal scenes     |
| Street    | Roads, pavements, and streetscapes |

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
├── angelah_model.pth         # Saved PyTorch model weights
├── angelah_model.keras       # Saved TensorFlow model
├── runtime.txt               # Specifies Python version for deployment
├── templates/
│   └── index.html            # Two-screen web interface (welcome + classifier)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Dependencies

Python **3.9 or higher** is recommended.

| Package      | Purpose                                  |
| ------------ | ---------------------------------------- |
| flask        | Web server and routing                   |
| torch        | PyTorch model training and inference     |
| torchvision  | Image transforms and dataset utilities   |
| tensorflow   | TensorFlow model training and inference  |
| Pillow       | Image loading and preprocessing in Flask |
| scikit-learn | Classification metrics                   |
| numpy        | Array operations                         |
| matplotlib   | Plotting                                 |
| seaborn      | Confusion matrix heatmaps                |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/angelahgaolatlhe/Intel_Image_Classification.git
cd Intel_Image_Classification
```

---

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:**
> Default installations use CPU. For NVIDIA GPUs:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

TensorFlow automatically uses GPU if CUDA is configured.

---

## Dataset Setup

1. Download the dataset from Kaggle:
   [https://www.kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

2. Extract into a `data/` folder:

```
Intel_Image_Classification/
└── data/
    ├── seg_train/
    └── seg_test/
```

---

## Training

Train models using `main.py`:

**PyTorch:**

```bash
python main.py --framework pytorch
```

**TensorFlow:**

```bash
python main.py --framework tensorflow
```

Outputs:

* `angelah_model.pth`
* `angelah_model.keras`

---

## Evaluation

**PyTorch:**

```bash
python pytorch_evaluation.py
```

**TensorFlow:**

```bash
python tensorflow_evaluation.py
```

Outputs include:

* Accuracy
* Classification report
* Confusion matrix

---

## Running the Web Application (Local)

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## Deployment on Render

This project is deployed using Render.

### 1. Push your project to GitHub

Repository:
[https://github.com/angelahgaolatlhe/Intel_Image_Classification](https://github.com/angelahgaolatlhe/Intel_Image_Classification)

---

### 2. Add a Python version file (IMPORTANT)

Create a file named:

```
runtime.txt
```

Add:

```
python-3.9.18
```

---

### 3. Update `app.py` for production

```python
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
```

---

### 4. Create a Web Service on Render

1. Go to [https://render.com](https://render.com)
2. Click **New + → Web Service**
3. Connect GitHub
4. Select your repository

---

### 5. Configure settings

* **Environment:** Python 3
* **Build Command:**

```bash
pip install -r requirements.txt
```

* **Start Command:**

```bash
python app.py
```

---

### 6. Ensure model files exist

* `angelah_model.pth`
* `angelah_model.keras`

---

### 7. Deploy

Click **Create Web Service** and wait for deployment.

Your app will be live at:

```
https://your-app-name.onrender.com
```

---

### Notes

* Free tier may sleep after inactivity
* No GPU support (CPU only)
* Ensure correct file paths for models

---

## Author

**Angelah Kgato Gaolatlhe**

---

## License

This project is for academic purposes.

