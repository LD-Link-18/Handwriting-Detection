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

device = torch.device("cpu")
IMG_WIDTH = 128
IMG_HEIGHT = 32
TEST_SAMPLE_SIZE = 500 
BEAM_SIZE = 3 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORDS_TXT_PATH = os.path.join(BASE_DIR, "ascii", "words.txt")
IAM_LINES_PATH = os.path.join(BASE_DIR, "ascii", "lines.txt")
WORDS_IMG_DIR = r"C:\Users\dikil\Desktop\crnn\words\iam_dataset\words"
MODEL_PATH = "words_model.pth"

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
    print("Loading Trigram Data...")
    words = []
    if not os.path.exists(lines_path):
        print(f"Warning: {lines_path} not found")
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

def smart_autocorrect_ultra(word, trigrams, bigrams, unigrams, spell, prev_word=None, prev_prev_word=None):
    word = word.lower()
    if not word.isalpha() or len(word) < 2: return word
    if word in unigrams and unigrams[word] > 80: return word
    candidates = spell.candidates(word)
    if not candidates: return word
    best_cand, max_score = word, -1
    for cand in candidates:
        score = unigrams.get(cand, 0)
        if prev_word: score += bigrams.get((prev_word, cand), 0) * 40
        if prev_word and prev_prev_word: score += trigrams.get((prev_prev_word, prev_word, cand), 0) * 100
        if score > max_score:
            max_score, best_cand = score, cand
    return best_cand

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
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print(f"Error: Model file {MODEL_PATH} not found")
        exit()

    TRIGRAMS, BIGRAMS, UNIGRAMS = load_ultra_ngram_data(IAM_LINES_PATH)
    spell = SpellChecker()

    if not os.path.exists(WORDS_TXT_PATH):
        print(f"Error: {WORDS_TXT_PATH} not found")
        exit()
        
    words_list = []
    with open(WORDS_TXT_PATH, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.split()
            if parts[1] == "ok":
                words_list.append((parts[0], parts[-1]))

    test_samples = random.sample(words_list, min(len(words_list), TEST_SAMPLE_SIZE))
    actual_texts, predicted_texts = [], []
    exact_match_count = 0
    prev_label, prev_prev_label = None, None

    print(f"Starting test for {len(test_samples)} words")

    for idx, (img_id, target_label) in enumerate(test_samples):
        p = img_id.split('-')
        img_path = os.path.join(WORDS_IMG_DIR, p[0], f"{p[0]}-{p[1]}", f"{img_id}.png")
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_tensor = torch.FloatTensor(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_tensor).log_softmax(2).permute(1, 0, 2)
            raw_pred = decode_beam_search(preds, num_to_char, BEAM_SIZE)
            final_pred = smart_autocorrect_ultra(raw_pred, TRIGRAMS, BIGRAMS, UNIGRAMS, spell, prev_label, prev_prev_label)
            
        actual_texts.append(target_label.lower())
        predicted_texts.append(final_pred.lower())
        
        if target_label.lower() == final_pred.lower():
            exact_match_count += 1
        
        prev_prev_label, prev_label = prev_label, final_pred

        if (idx+1) % 100 == 0:
            print(f"Processed: {idx+1}/{len(test_samples)}")

    if len(actual_texts) > 0:
        final_cer = cer(actual_texts, predicted_texts)
        final_wer = wer(actual_texts, predicted_texts)
        
        print("\n" + "="*50)
        print("FINAL EVALUATION REPORT")
        print("="*50)
        print(f"Samples Successfully Tested  : {len(actual_texts)}")
        print(f"Exact Match Accuracy         : {(exact_match_count/len(actual_texts))*100:.2f}%")
        print(f"Word Accuracy (1-WER)        : {(1-final_wer)*100:.2f}%")
        print(f"Character Accuracy (1-CER)   : {(1-final_cer)*100:.2f}%")
        print("-" * 50)
        print(f"Word Error Rate (WER)        : {final_wer * 100:.2f}%")
        print(f"Character Error Rate (CER)   : {final_cer * 100:.2f}%")
        print("="*50)
    else:
        print("\nError: No images were found at:", WORDS_IMG_DIR)