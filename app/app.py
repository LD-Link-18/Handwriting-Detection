import os
import re
import collections
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from spellchecker import SpellChecker

app = FastAPI(title="Handwriting & Font Recognizer API")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Running backend on device: {device.type.upper()}")

# ============================ 1. CONFIGURATION & ARCHITECTURES ============================

ALPHABET_PATH = "ocr/alphabet.txt"
TEXT_MODEL_PATH = "ocr/lines_model.pth"
WORDS_MODEL_PATH = "ocr/words_model.pth"
FONT_MODEL_PATH = "font_detection/font_recognition_model.pth"
TEXT_SOURCE_PATH = "ingilizce_metin.txt"

# Default fallback alphabet if file not found
DEFAULT_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#&'()*+,-./:;? "

if os.path.exists(ALPHABET_PATH):
    with open(ALPHABET_PATH, "r", encoding="utf-8") as f:
        alphabet = f.read()
else:
    alphabet = DEFAULT_ALPHABET

num_to_char = {i + 1: char for i, char in enumerate(alphabet)}
num_chars = len(alphabet)

FONT_CLASSES = [
    "Caveat-VariableFont_wght",
    "CourierPrime-Italic",
    "Lobster-Regular",
    "Merriweather-Italic-VariableFont_opsz,wdth,wght",
    "Roboto-Italic-VariableFont_wdth,wght"
]

FONT_TO_WRITER = {
    "Caveat-VariableFont_wght": "yabanci_yazar",
    "CourierPrime-Italic": "yazar1",
    "Lobster-Regular": "yazar2",
    "Merriweather-Italic-VariableFont_opsz,wdth,wght": "yazar3",
    "Roboto-Italic-VariableFont_wdth,wght": "yazar4",
}

# Translate writer labels for UI presentation
WRITER_DISPLAY_NAME = {
    "yabanci_yazar": "Foreign Writer (Unknown)",
    "yazar1": "Writer 1 (Courier style)",
    "yazar2": "Writer 2 (Lobster Regular style)",
    "yazar3": "Writer 3 (Merriweather style)",
    "yazar4": "Writer 4 (Roboto style)",
}


class CRNN(nn.Module):
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_chars + 1)

    def forward(self, x):
        x = self.cnn(x).permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.rnn(x)
        return self.fc(x)


def get_resnet18_model(num_classes):
    # Load ResNet-18 without pre-trained weights since we load state_dict directly
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

# Load Models
try:
    htr_lines_model = CRNN(num_chars).to(device)
    htr_lines_model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device))
    htr_lines_model.eval()
    print("[SUCCESS] Loaded HTR Lines Model successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load HTR lines model: {e}")
    htr_lines_model = None

try:
    htr_words_model = CRNN(num_chars).to(device)
    htr_words_model.load_state_dict(torch.load(WORDS_MODEL_PATH, map_location=device))
    htr_words_model.eval()
    print("[SUCCESS] Loaded HTR Words Model successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load HTR words model: {e}")
    htr_words_model = None

try:
    font_model = get_resnet18_model(len(FONT_CLASSES)).to(device)
    font_model.load_state_dict(torch.load(FONT_MODEL_PATH, map_location=device))
    font_model.eval()
    print("[SUCCESS] Loaded Font Recognition Model successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load Font model: {e}")
    font_model = None

# ============================ 2. LANGUAGE MODEL & SPELL CHECKER ============================

spell = SpellChecker()

def load_ngrams(text_path):
    words = []
    if os.path.exists(text_path):
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Clean words and extract alphabetic sequences
                    text = line.lower().replace("|", " ")
                    clean_text = re.sub(r'[^a-z\s]', '', text)
                    words.extend(clean_text.split())
        except Exception as e:
            print(f"[ERROR] Loading language model data: {e}")
    
    trigrams = collections.Counter(zip(words, words[1:], words[2:]))
    bigrams = collections.Counter(zip(words, words[1:]))
    unigrams = collections.Counter(words)
    return trigrams, bigrams, unigrams

TRIGRAMS, BIGRAMS, UNIGRAMS = load_ngrams(TEXT_SOURCE_PATH)
print(f"[SYSTEM] Loaded {len(UNIGRAMS)} unigrams from {TEXT_SOURCE_PATH} for spelling correction.")


def correct_sentence(sentence, trigrams, bigrams, unigrams, spell_chk):
    words = sentence.split()
    corrected_words = []
    prev_word = None
    prev_prev_word = None

    for word in words:
        word = word.lower()
        # Do not correct short, non-alphabetic, or very common words
        if not word.isalpha() or len(word) < 2:
            corrected_words.append(word)
        elif word in unigrams and unigrams[word] > 5:  # lowered threshold from 80 because dataset size is smaller
            corrected_words.append(word)
        else:
            candidates = spell_chk.candidates(word)
            if not candidates:
                corrected_words.append(word)
            else:
                best_cand, max_score = word, -1
                for cand in candidates:
                    score = unigrams.get(cand, 0)
                    if prev_word:
                        score += bigrams.get((prev_word, cand), 0) * 40
                    if prev_word and prev_prev_word:
                        score += trigrams.get((prev_prev_word, prev_word, cand), 0) * 100
                    if score > max_score:
                        max_score, best_cand = score, cand
                corrected_words.append(best_cand)

        prev_prev_word = prev_word
        prev_word = corrected_words[-1]

    return " ".join(corrected_words)


# ============================ 3. IMAGE PREPROCESSING ============================

def preprocess_htr(image_bytes, is_word_mode=False):
    # Load grayscale image from memory
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image format.")

    # Direct resize to target dimensions (no CLAHE/binarization/padding)
    target_w = 128 if is_word_mode else 512
    img = cv2.resize(img, (target_w, 32))

    # Convert to Tensor normalized to [0, 1]
    img_tensor = torch.FloatTensor(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    return img_tensor


def resize_and_pad_pil(image, output_size=224):
    # Keep aspect ratio and fit inside output_size
    image.thumbnail((output_size, output_size), Image.Resampling.LANCZOS)
    
    # Create white canvas and center the resized image
    final = Image.new("RGB", (output_size, output_size), color="white")
    x = (output_size - image.width) // 2
    y = (output_size - image.height) // 2
    final.paste(image, (x, y))
    return final


def preprocess_font(image_bytes):
    # Load color image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise ValueError("Invalid image format.")
    
    # 1. White-point stretching to clean up background to pure white while preserving typographic details
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Find the 90th percentile of brightness in grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    p90 = float(np.percentile(gray, 90))
    if p90 < 1.0:
        p90 = 1.0
        
    # Scale image to make background pure white, clipping output to [0, 255]
    img_stretched = np.clip((img_rgb.astype(np.float32) / p90) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(img_stretched)

    # Scale directly to 224x224 resolution
    image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)

    # Normalization transforms
    font_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = font_transform(image_resized).unsqueeze(0).to(device)
    return img_tensor


# ============================ 4. INFERENCE DECODERS ============================

def decode_beam_search(preds, num_to_char, beam_size=3):
    # preds is [seq_len, batch_size, num_classes] -> get first element of batch
    preds = preds[:, 0, :]
    paths = [(0.0, -1, [])]

    for t in range(preds.size(0)):
        step_probs = preds[t]
        topk_logprobs, topk_indices = torch.topk(step_probs, beam_size)

        new_paths = []
        for score, prev_idx, char_list in paths:
            for i in range(beam_size):
                idx = topk_indices[i].item()
                logp = topk_logprobs[i].item()

                new_score = score + logp
                new_char_list = list(char_list)

                # Blank token is 0 in CTC
                if idx != 0 and idx != prev_idx:
                    char = num_to_char.get(idx, "")
                    if char:
                        new_char_list.append(char)

                new_paths.append((new_score, idx, new_char_list))

        new_paths.sort(key=lambda x: x[0], reverse=True)
        paths = new_paths[:beam_size]

    best_path = paths[0][2]
    return "".join(best_path)


# ============================ 5. API ENDPOINTS ============================

# Ensure directories exist
os.makedirs("app/templates", exist_ok=True)
os.makedirs("app/static", exist_ok=True)

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/api/predict-text")
async def predict_text(file: UploadFile = File(...), text_type: str = Form("line")):
    is_word = (text_type == "word")
    model_to_use = htr_words_model if is_word else htr_lines_model
    
    if model_to_use is None:
        raise HTTPException(status_code=500, detail="Requested HTR Model is not loaded on server.")
    
    try:
        contents = await file.read()
        img_tensor = preprocess_htr(contents, is_word_mode=is_word)
        
        with torch.no_grad():
            preds = model_to_use(img_tensor).log_softmax(2).permute(1, 0, 2)
            raw_text = decode_beam_search(preds, num_to_char, beam_size=3)
            
        corrected_text = correct_sentence(raw_text, TRIGRAMS, BIGRAMS, UNIGRAMS, spell)
        
        return {
            "success": True,
            "raw_text": raw_text,
            "corrected_text": corrected_text
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")


@app.post("/api/predict-font")
async def predict_font(file: UploadFile = File(...)):
    if font_model is None:
        raise HTTPException(status_code=500, detail="Font recognition model is not loaded on server.")
        
    try:
        contents = await file.read()
        img_tensor = preprocess_font(contents)
        
        with torch.no_grad():
            output = font_model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            
        probs_np = probs.cpu().numpy()
        sorted_idx = np.argsort(probs_np)[::-1]
        
        predicted_font = FONT_CLASSES[sorted_idx[0]]
        confidence = float(probs_np[sorted_idx[0]])
        
        writer = FONT_TO_WRITER.get(predicted_font, predicted_font)
        writer_display = WRITER_DISPLAY_NAME.get(writer, writer)
        
        # Foreign writer condition
        if predicted_font == "Caveat-VariableFont_wght":
            message = "Bu yazar bizim yazarlarımızdan değil (Unknown Writer)"
        else:
            message = f"Detected: {writer_display}"
            
        # Get scores for all classes
        scores = []
        for idx in sorted_idx:
            font_cls = FONT_CLASSES[idx]
            w_name = FONT_TO_WRITER.get(font_cls, font_cls)
            w_disp = WRITER_DISPLAY_NAME.get(w_name, w_name)
            scores.append({
                "font": font_cls,
                "writer": w_name,
                "writer_display": w_disp,
                "probability": float(probs_np[idx])
            })
            
        return {
            "success": True,
            "predicted_font": predicted_font,
            "writer": writer,
            "writer_display": writer_display,
            "confidence": confidence,
            "message": message,
            "scores": scores
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")


@app.post("/api/predict-all")
async def predict_all(file: UploadFile = File(...), text_type: str = Form("line")):
    is_word = (text_type == "word")
    # Run both models on the same image
    contents = await file.read()
    
    # HTR Predict
    htr_result = {"success": False, "error": "HTR Model not loaded"}
    model_to_use = htr_words_model if is_word else htr_lines_model
    if model_to_use is not None:
        try:
            img_tensor_htr = preprocess_htr(contents, is_word_mode=is_word)
            with torch.no_grad():
                preds = model_to_use(img_tensor_htr).log_softmax(2).permute(1, 0, 2)
                raw_text = decode_beam_search(preds, num_to_char, beam_size=3)
            corrected_text = correct_sentence(raw_text, TRIGRAMS, BIGRAMS, UNIGRAMS, spell)
            htr_result = {
                "success": True,
                "raw_text": raw_text,
                "corrected_text": corrected_text
            }
        except Exception as e:
            htr_result = {"success": False, "error": str(e)}
            
    # Font Predict
    font_result = {"success": False, "error": "Font model not loaded"}
    if font_model is not None:
        try:
            img_tensor_font = preprocess_font(contents)
            with torch.no_grad():
                output = font_model(img_tensor_font)
                probs = torch.softmax(output, dim=1)[0]
            
            probs_np = probs.cpu().numpy()
            sorted_idx = np.argsort(probs_np)[::-1]
            
            predicted_font = FONT_CLASSES[sorted_idx[0]]
            confidence = float(probs_np[sorted_idx[0]])
            writer = FONT_TO_WRITER.get(predicted_font, predicted_font)
            writer_display = WRITER_DISPLAY_NAME.get(writer, writer)
            
            if predicted_font == "Caveat-VariableFont_wght":
                message = "Bu yazar bizim yazarlarımızdan değil (Unknown Writer)"
            else:
                message = f"Detected: {writer_display}"
                
            scores = []
            for idx in sorted_idx:
                font_cls = FONT_CLASSES[idx]
                w_name = FONT_TO_WRITER.get(font_cls, font_cls)
                w_disp = WRITER_DISPLAY_NAME.get(w_name, w_name)
                scores.append({
                    "font": font_cls,
                    "writer": w_name,
                    "writer_display": w_disp,
                    "probability": float(probs_np[idx])
                })
                
            font_result = {
                "success": True,
                "predicted_font": predicted_font,
                "writer": writer,
                "writer_display": writer_display,
                "confidence": confidence,
                "message": message,
                "scores": scores
            }
        except Exception as e:
            font_result = {"success": False, "error": str(e)}
            
    return {
        "text_recognition": htr_result,
        "font_recognition": font_result
    }
