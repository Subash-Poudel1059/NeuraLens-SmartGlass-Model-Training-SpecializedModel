# NeuraLens SmartGlass Model Training (Specialized Model)

![alt text](https://img.shields.io/badge/license-MIT-blue.svg)
![alt text](https://img.shields.io/badge/python-3.8%2B-green.svg)
![alt text](https://img.shields.io/badge/status-active-orange.svg)

This repository is a specialized module of the NeuraLens Offline ecosystem. It focuses on training, fine-tuning, and optimizing lightweight neural network models specifically designed to run locally on the NeuraLens SmartGlass hardware without requiring an internet connection.

## 🚀 Overview
The Specialized Model Training repo provides the pipeline to create "Edge-Ready" models. Because the SmartGlass operates offline, this module emphasizes model compression, quantization (TFLite/ONNX), and high-accuracy detection of specific indoor objects that are critical for the visually impaired.

## ✨ Key Features
* **Custom Dataset Integration:** Tools to preprocess and label indoor-specific data (e.g., specific household items, medicine bottles, or indoor landmarks).
* **Transfer Learning Pipeline:** Fine-tune state-of-the-art lightweight architectures like YOLOv8-tiny, MobileNetV3, or EfficientNet-Edge.
* **Quantization & Optimization:** Automated scripts to convert trained models into optimized formats for edge devices to ensure high FPS with low battery consumption.
* **Accuracy Benchmarking:** Built-in evaluation scripts to test model performance in low-light and blurred-motion scenarios typical of wearable cameras.

## 🛠 Tech Stack
* **Language:** Python
* **AI Frameworks:** PyTorch / TensorFlow / Ultralytics
* **Optimization:** ONNX Runtime, TFLite Converter
* **Data Processing:** OpenCV, NumPy, Albumentations (for data augmentation)
* **Hardware Targeting:** Optimization for ARM-based processors / Edge TPUs

## 📦 Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Subash-Poudel1059/NeuraLens-SmartGlass-Model-Training-SpecializedModel
cd NeuraLens-SmartGlass-Model-Training-SpecializedModel
```

**2. Create a virtual environment:**
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Environment variables:**
Create a `.env` file in the root directory to manage dataset paths and training logs:
```env
DATASET_PATH=./data/custom_dataset
OUTPUT_MODEL_PATH=./models/exported/
EPOCHS=100
BATCH_SIZE=16
```

## 🧩 Classes
The default dataset in this repo includes 20 classes defined in `data.yaml`:
- bag
- bin
- bottle
- cctv_camera
- chair
- copy
- curtain
- desk
- door
- glass
- jug
- light
- pen
- person
- plants
- poster
- smartboard
- stairs
- switch
- watch

## 🚀 Usage

To start training the specialized model:
```bash
python train.py --config config/model_config.yaml --device cuda
```

To export the trained model for offline SmartGlass use:
```bash
python export.py --model last.pt --format tflite
```

- `--config`: Path to the training configuration file.
- `--format`: The target format for the offline edge device (`tflite`, `onnx`, or `engine`).

