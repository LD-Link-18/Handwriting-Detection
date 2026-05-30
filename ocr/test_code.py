import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import collections
import re
from spellchecker import SpellChecker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[SİSTEM BİLGİSİ] Kullanılan Cihaz: {device.type.upper()}")


MODEL_DOSYASI = "lines_model.pth"  
IMG_WIDTH = 512
IMG_HEIGHT = 32
BEAM_SIZE = 3 


test_image_path = r"C:\Users\dikil\Desktop\crnn\fotolar\device.png"
IAM_LINES_PATH = r"C:\Users\dikil\Desktop\crnn\ascii\lines.txt"

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

def decode_beam_search(preds, num_to_char, beam_size=3):
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
                
                if idx != 0 and idx != prev_idx:
                    new_char_list.append(num_to_char[idx])
                
                new_paths.append((new_score, idx, new_char_list))
                
        new_paths.sort(key=lambda x: x[0], reverse=True)
        paths = new_paths[:beam_size]
        
    best_path = paths[0][2]
    return "".join(best_path)

def resize_with_pad(img, target_w, target_h):
    
    h, w = img.shape
    ratio = w / h
    target_ratio = target_w / target_h
    
    if ratio > target_ratio:
        new_w = target_w
        new_h = int(target_w / ratio)
    else:
        new_h = target_h
        new_w = int(target_h * ratio)
        
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, 0:new_w] = resized
    return canvas

def load_ultra_ngram_data(lines_path):
    words = []
    if not os.path.exists(lines_path):
        return {}, {}, {}
    try:
        with open(lines_path, 'r') as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
                parts = line.split()
                text = " ".join(parts[8:]).lower().replace("|", " ")
                clean_text = re.sub(r'[^a-z\s]', '', text)
                words.extend(clean_text.split())
        trigrams = collections.Counter(zip(words, words[1:], words[2:]))
        bigrams = collections.Counter(zip(words, words[1:]))
        unigrams = collections.Counter(words)
        return trigrams, bigrams, unigrams
    except:
        return {}, {}, {}

def correct_sentence(sentence, trigrams, bigrams, unigrams, spell):
    words = sentence.split()
    corrected_words = []
    prev_word, prev_prev_word = None, None
    
    for word in words:
        word = word.lower()
        if not word.isalpha() or len(word) < 2:
            corrected_words.append(word)
        elif word in unigrams and unigrams[word] > 80:
            corrected_words.append(word)
        else:
            candidates = spell.candidates(word)
            if not candidates:
                corrected_words.append(word)
            else:
                best_cand, max_score = word, -1
                for cand in candidates:
                    score = unigrams.get(cand, 0)
                    if prev_word: score += bigrams.get((prev_word, cand), 0) * 40
                    if prev_word and prev_prev_word: score += trigrams.get((prev_prev_word, prev_word, cand), 0) * 100
                    if score > max_score:
                        max_score, best_cand = score, cand
                corrected_words.append(best_cand)
                
        prev_prev_word = prev_word
        prev_word = corrected_words[-1]
        
    return " ".join(corrected_words)

if __name__ == "__main__":
    
    if os.path.exists('alphabet.txt'):
        with open('alphabet.txt', 'r', encoding='utf-8') as f: 
            alphabet = f.read()
    else:
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-() "
        
    num_to_char = {i + 1: char for i, char in enumerate(alphabet)}
    
    
    model = CRNN(len(alphabet)).to(device)
    if os.path.exists(MODEL_DOSYASI):
        model.load_state_dict(torch.load(MODEL_DOSYASI, map_location=device))
        model.eval()
        print(f"Model başarıyla yüklendi: {MODEL_DOSYASI}")
    else:
        print(f"HATA: Model dosyası '{MODEL_DOSYASI}' bulunamadı.")
        exit()

    
    TRIGRAMS, BIGRAMS, UNIGRAMS = load_ultra_ngram_data(IAM_LINES_PATH)
    spell = SpellChecker()

    if not os.path.exists(test_image_path):
        print(f"HATA: Test resmi '{test_image_path}' bulunamadı.")
        exit()

   
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    
  
    img = resize_with_pad(img, IMG_WIDTH, IMG_HEIGHT)
    
    
    img_tensor = torch.FloatTensor(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

   
    with torch.no_grad():
        preds = model(img_tensor).log_softmax(2).permute(1, 0, 2)
        raw_text = decode_beam_search(preds, num_to_char, BEAM_SIZE)

    
    final_text = correct_sentence(raw_text, TRIGRAMS, BIGRAMS, UNIGRAMS, spell)

    print("\n" + ""*25)
    print("           TAHMİN SONUCU")
    print(""*25)
    print(f"Optik Okuma (Ham)   : {raw_text}")
    print(f"Yapay Zeka Düzeltili: {final_text}")
    print(""*25 + "\n")