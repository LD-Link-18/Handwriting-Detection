# ScribbleMind: Real-time Handwriting OCR & Font Style Identification

ScribbleMind is a web application that implements a dual-model deep learning pipeline for:
1. **Handwriting Optical Character Recognition (OCR)**: Transcribes images of handwritten lines or single words into digital text.
2. **Font & Writer Identification**: Identifies the font style or writer characteristics of handwritten text using image classification.

The project features a FastAPI-based backend and a responsive front-end dashboard designed with modern dark-mode glassmorphic aesthetics, webcam/camera support, interactive preset selection, and real-time visualization of neural network confidence distributions.

---

## 📂 Project Structure

The codebase is organized into modular directories:

```bash
ScribbleMind/
│
├── ocr/                     # Handwriting Text Recognition (HTR) component
│   ├── lines_model.pth      # CRNN weights for sentence-line OCR (512x32)
│   ├── words_model.pth      # CRNN weights for single-word OCR (128x32)
│   ├── alphabet.txt         # Character vocabulary list
│   ├── train_lines.py       # Training scripts for HTR models
│   └── evaluate_lines.py    # Testing/evaluation metrics (WER/CER)
│
├── font_detection/          # Font & Writer classification component
│   ├── fontt_model_tam.pth  # ResNet-18 full model object weights
│   ├── font_tanima.py       # Training script with L2 Regularization
│   └── dataset_images/      # Sub-divided train/val/test splits (224x224)
│
├── app/                     # FastAPI backend & web server assets
│   ├── app.py               # Main API endpoints controller
│   ├── templates/
│   │   └── index.html       # Single-page front-end application
│   └── static/
│       └── example_*.png    # Test preset images for user demonstrations
│
├── ingilizce_metin.txt      # Source text corpus used for dataset and spellcheck
└── README.md                # Project documentation
```

---

## ⚡ Features

* **High-Accuracy HTR**: Powered by a **Convolutional Recurrent Neural Network (CRNN)** combined with **CTC Beam Search decoding** and trigram n-gram spelling auto-correction.
* **Generalizing Style Identification**: Classifies custom handwritten scripts into 5 writer/font families using a custom **ResNet-18 classifier** optimized for paper textures and strokes.
* **White-point Stretching**: Performs real-time preprocessing using the 90th percentile of brightness values to eliminate shadows and background noise without degrading thin pencil strokes.
* **Responsive HTML5 Camera APIs**: Capture snapshots directly from front/back cameras on mobile devices or webcams on desktops.
* **Modular Backend Architecture**: Serving separate, specialized OCR and Font API endpoints, plus a combined All-In-One pipeline for concurrent prediction.

---

## 🛠️ Setup & Installation

Follow these steps to configure your local Python environment:

### 1. Clone & Set Up Directory
Open your terminal in the project directory `/home/samet/Projects/Handwriting-Detection`.

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required packages in your virtual environment:
```bash
pip install torch torchvision opencv-python-headless fastapi uvicorn python-multipart jinja2 pyspellchecker numpy pillow scikit-learn
```

---

## 🚀 Running the Server

Start the FastAPI application with Uvicorn:
```bash
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
```

Once the server successfully loads the deep learning models and reports `[SYSTEM] Running backend on device: CUDA` (if GPU is available), open the application in your browser:
* **Desktop Browser**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Mobile Browser**: `http://<your-local-ip>:8000` (e.g., `http://192.168.1.18:8000` if connected to the same Wi-Fi).

---

## 📡 API Documentation

### 1. Read Text (OCR)
* **Endpoint**: `/api/predict-text`
* **Method**: `POST`
* **Form Parameters**:
  * `file`: Binary image file (JPEG/PNG)
  * `text_type`: `"line"` (Sentence line) or `"word"` (Single word)
* **Response**:
  ```json
  {
    "success": true,
    "raw_text": "hello wor1d",
    "corrected_text": "hello world"
  }
  ```

### 2. Classify Font & Writer Style
* **Endpoint**: `/api/predict-font`
* **Method**: `POST`
* **Form Parameters**:
  * `file`: Binary image file (JPEG/PNG)
* **Response**:
  ```json
  {
    "success": true,
    "predicted_font": "Caveat-VariableFont_wght",
    "writer_display": "Foreign Writer (Unknown)",
    "message": "Bu yazar bizim yazarlarımızdan değil (Unknown Writer)",
    "confidence": 0.9999,
    "scores": [
      { "font": "Caveat-VariableFont_wght", "writer_display": "Foreign Writer", "probability": 0.9999 },
      ...
    ]
  }
  ```

### 3. Dual Processing (All-In-One)
* **Endpoint**: `/api/predict-all`
* **Method**: `POST`
* **Form Parameters**:
  * `file`: Binary image file (JPEG/PNG)
  * `text_type`: `"line"` or `"word"`
* **Response**:
  ```json
  {
    "text_recognition": { "success": true, "raw_text": "...", "corrected_text": "..." },
    "font_recognition": { "success": true, "predicted_font": "...", "confidence": 0.99, ... }
  }
  ```

---

## 🎓 Model Training

To retrain the writer style classifier on the `ingilizce_metin.txt` corpus with updated fonts:
1. Ensure the TrueType Font files (`.ttf`) are placed under `font_detection/Fonts`.
2. Run the dataset generation and training sequence:
   ```bash
   python font_detection/font_tanima.py
   ```
This script renders the corpus words, applies image data augmentation (rotation, perspective shift, blur, noise), trains the ResNet-18 parameters, and saves the final checkpoint to `font_detection/fontt_model_tam.pth`.