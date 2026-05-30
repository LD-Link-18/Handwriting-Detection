import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class IAMWordsDataset(Dataset):
    def __init__(self, base_path, words_txt_path, alphabet, img_height=32, img_width=128):
        self.base_path = base_path
        self.img_height = img_height
        self.img_width = img_width
        self.data = []
        
        self.char2idx = {char: idx + 1 for idx, char in enumerate(alphabet)}
        self.char2idx['[BLANK]'] = 0 
        
        print(f"Scanning IAM Words data: {words_txt_path}")
        with open(words_txt_path, 'r') as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
                parts = line.split()
                
                if parts[1] == "ok":
                    img_id = parts[0]
                    label = parts[-1].lower()
                    
                    p = img_id.split('-')
                    img_path = os.path.join(base_path, "words", p[0], f"{p[0]}-{p[1]}", f"{img_id}.png")
                    
                    if os.path.exists(img_path):
                        self.data.append((img_path, label))

        print(f"Total valid word images found: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize((self.img_width, self.img_height), Image.Resampling.BILINEAR)
            
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0) 

            target = [self.char2idx.get(c, 0) for c in label]
            target_tensor = torch.tensor(target, dtype=torch.long)
            
            return img_tensor, target_tensor, label
            
        except Exception:
            import random
            random_idx = random.randint(0, len(self.data) - 1)
            return self.__getitem__(random_idx)

if __name__ == "__main__":
    BASE_PATH = r"C:\Users\dikil\Desktop\proje"
    WORDS_TXT = os.path.join(BASE_PATH, "ascii", "words.txt")
    ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-()"
    
    dataset = IAMWordsDataset(BASE_PATH, WORDS_TXT, ALPHABET)
    if len(dataset) > 0:
        img, target, label = dataset[0]
        print("\nWords Dataset Test Successful!")
        print(f"Image Shape: {img.shape}")
        print(f"Text: '{label}' -> Tokens: {target.tolist()}")