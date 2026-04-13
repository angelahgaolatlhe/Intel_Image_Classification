# Intel Image Classification Project

This project implements a complete image classification pipeline using the Intel Image Classification dataset. It includes:

* Data preprocessing and augmentation
* CNN models built with PyTorch and TensorFlow
* Command-line training interface
* A web application for real-time image classification

---

## Dataset

Dataset used: Intel Image Classification

URL: https://www.kaggle.com/datasets/puneet6060/intel-image-classification 

Classes:

* Buildings
* Forest
* Glacier
* Mountain
* Sea
* Street

---

## Project Structure

```
intel_classifier_project/
│── data/
│   ├── seg_train/
│   ├── seg_test/
│   └── seg_pred/
│
│── pytorch_model.py
│── tensorflow_model.py
│── main.py
│── app.py
│── templates/
│── static/
│── requirements.txt
│── README.md
```

---

## Installation

1. Clone the repository

```
git clone <your-repo-url>
cd intel_classifier_project
```

2. Install dependencies

```
pip install -r requirements.txt
```

---

## Training Models

Train using command-line argument:

### PyTorch

```
python main.py --framework pytorch
```

### TensorFlow

```
python main.py --framework tensorflow
```

---

## Saved Models

After training:

* PyTorch → `yourname_model.pth`
* TensorFlow → `yourname_model.keras`

---

# Web Application

## Run the Flask app

```
python app.py
```

## Features

* Select model (PyTorch / TensorFlow)
* Upload image
* View prediction instantly

Open browser:

```
http://localhost:5000
```

---

# Model Details

## Input

* Image size: 224 × 224 × 3

## Architecture

* Convolutional Neural Networks (CNN)
* ReLU activation
* MaxPooling
* Fully connected layers
* Dropout for regularization

## Output

* 6-class classification

---

# Improvements (Future Work)

* Transfer learning (ResNet, MobileNet)
* Hyperparameter tuning
* Model evaluation metrics (confusion matrix)
* Deployment (Docker / Cloud)

---

# Author

Angelah Kgato Gaolatlhe

---

# License

This project is for academic purposes.
