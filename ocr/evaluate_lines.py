import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import random
import collections
import re
from jiwer import cer, wer
from spellchecker import SpellChecker 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[TEST BİLGİSİ] Kullanılan Cihaz: {device.type.upper()}")

IMG_WIDTH = 512
IMG_HEIGHT = 32
TEST_SAMPLE_SIZE = 500 
BEAM_SIZE = 3


IAM_LINES_PATH = r"C:\Users\dikil\Desktop\crnn\ascii\lines.txt"
LINES_IMG_DIR = r"C:\Users\dikil\Desktop\crnn\lines"
MODEL_PATH = "lines_model.pth" 

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
    return "".join(paths[0][2])

def load_ultra_ngram_data(lines_path):
    print("Trigram/Bigram Dil Modeli Yükleniyor...")
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
    prev_word = None
    prev_prev_word = None
    
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
        with open('alphabet.txt', 'r', encoding='utf-8') as f: alphabet = f.read()
    else:
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-() "
    num_to_char = {i + 1: char for i, char in enumerate(alphabet)}

    model = CRNN(len(alphabet)).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"Model başarıyla yüklendi: {MODEL_PATH}")
    else:
        print(f"HATA: '{MODEL_PATH}' bulunamadı. Lütfen önce train_lines.py eğitimini bitir.")
        exit()

    TRIGRAMS, BIGRAMS, UNIGRAMS = load_ultra_ngram_data(IAM_LINES_PATH)
    spell = SpellChecker()

    if not os.path.exists(IAM_LINES_PATH):
        print(f"HATA: {IAM_LINES_PATH} bulunamadı")
        exit()
        
    lines_list = []
    with open(IAM_LINES_PATH, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            if parts[1] == "ok":
                label = " ".join(parts[8:]).replace("|", " ")
                lines_list.append((parts[0], label))

    test_samples = random.sample(lines_list, min(len(lines_list), TEST_SAMPLE_SIZE))
    actual_texts, predicted_texts = [], []
    exact_match_count = 0

    print(f"\n{len(test_samples)} adet satır/cümle üzerinde GPU ile test başlatılıyor...\n")

    for idx, (img_id, target_label) in enumerate(test_samples):
        p = img_id.split('-')
        img_path = os.path.join(LINES_IMG_DIR, p[0], f"{p[0]}-{p[1]}", f"{img_id}.png")
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
       
        img_tensor = torch.FloatTensor(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_tensor).log_softmax(2).permute(1, 0, 2)
            raw_pred = decode_beam_search(preds, num_to_char, BEAM_SIZE)
            final_pred = correct_sentence(raw_pred, TRIGRAMS, BIGRAMS, UNIGRAMS, spell)
            
        actual_texts.append(target_label.lower())
        predicted_texts.append(final_pred.lower())
        
        if target_label.lower() == final_pred.lower():
            exact_match_count += 1

        if (idx+1) % 100 == 0:
            print(f"İşlenen: {idx+1}/{len(test_samples)}")

    if len(actual_texts) > 0:
        final_cer = cer(actual_texts, predicted_texts)
        final_wer = wer(actual_texts, predicted_texts)
        
        print("\n" + "="*50)
        print(" SATIR (CÜMLE) TANIMA NİHAİ TEST RAPORU ")
        print("="*50)
        print(f"Başarıyla Test Edilen Örnek Sayısı : {len(actual_texts)}")
        print(f"Tam Eşleşme (Exact Match) Oranı    : {(exact_match_count/len(actual_texts))*100:.2f}%")
        print(f"Cümle Kelime Doğruluğu (1-WER)     : {(1-final_wer)*100:.2f}%")
        print(f"Cümle Harf Doğruluğu (1-CER)       : {(1-final_cer)*100:.2f}%")
        print("-" * 50)
        print(f"Kelime Hata Oranı (WER)            : {final_wer * 100:.2f}%")
        print(f"Karakter Hata Oranı (CER)          : {final_cer * 100:.2f}%")
        print("="*50)
    else:
        print(f"\nHATA: Test edilecek resim bulunamadı. Lütfen '{LINES_IMG_DIR}' yolunu kontrol et.")