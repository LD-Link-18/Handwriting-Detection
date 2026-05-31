"""
FONT TANIMA - ResNet-50 (Overfitting Korumalı)
===============================================
"""

import os
import re
import random
import warnings
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from tqdm import tqdm
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# ============================ AYARLAR ============================
TXT_PATH = "ingilizce_metin.txt"
FONT_DIR = "fonts"
DATASET_DIR = "dataset_images"
MODEL_PATH = "font_recognition_model.pth"

# ============================ YAZAR EŞLEŞTİRME ============================
# Font dosya adı -> Yazar adı. Caveat = yabancı, diğer 4'ü = yerli yazarlar.
FONT_TO_WRITER = {
    "Caveat-VariableFont_wght": "yabanci_yazar",
    "CourierPrime-Italic": "yazar1",
    "Lobster-Regular": "yazar2",
    "Merriweather-Italic-VariableFont_opsz,wdth,wght": "yazar3",
    "Roboto-Italic-VariableFont_wdth,wght": "yazar4",
}

YABANCI_YAZAR_FONT = "Caveat-VariableFont_wght"  # Bu tahmin edilirse özel mesaj verilecek

IMG_SIZE = 400
FONT_SIZE = 72          # Variable fontlar için güvenli boyut
BATCH_SIZE = 4         # Küçük veri setinde büyük batch overfitting yapar
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4     # L2 regularizasyon (overfitting önler)
PATIENCE = 5            # Early stopping sabrı
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {DEVICE}")

def render_word_image(word, font_path, font_size=FONT_SIZE, max_size=400):
    """Tek kelimeyi render et, otomatik ölçeklendir."""
    try:
        font = ImageFont.truetype(str(font_path), size=font_size)
    except Exception as e:
        print(f"[UYARI] Font yüklenemedi: {font_path} -> {e}")
        return None

    # Kelime boyutunu ölç
    dummy = Image.new("RGB", (10, 10), color="white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Eğer kelime canvas'dan büyükse, font boyutunu küçült
    if text_w > max_size - 40 or text_h > max_size - 40:
        scale = min((max_size - 40) / text_w, (max_size - 40) / text_h)
        new_size = int(font_size * scale)
        font = ImageFont.truetype(str(font_path), size=new_size)
        # Tekrar ölç
        bbox = draw.textbbox((0, 0), word, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    # Canvas oluştur ve ortala
    img = Image.new("RGB", (max_size, max_size), color="white")
    draw = ImageDraw.Draw(img)
    x = (max_size - text_w) // 2
    y = (max_size - text_h) // 2
    draw.text((x, y - bbox[1]), word, font=font, fill="black")

    return img

# Seed sabitle (tekrarlanabilirlik)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================ 1. DATASET OLUŞTURMA ============================


def split_words(text):
    """Metni kelimelere ayır, noktalama işaretlerini at."""
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    # Tekrar eden kelimeleri çıkar, unique kelimeleri al
    unique_words = list(dict.fromkeys(words))
    # En az 3 harfli kelimeleri filtrele (a, I, vs. gibi kısa kelimeleri at)
    filtered = [w for w in unique_words if len(w) >= 3]
    return filtered

def augment_image(img):
    """Overfitting önleyici augmentasyonlar."""
    # 1. Hafif rotasyon
    if random.random() < 0.5:
        angle = random.uniform(-3.0, 3.0)
        img = img.rotate(angle, fillcolor="white", resample=Image.Resampling.BILINEAR)

    # 2. Hafif perspektif / shear
    if random.random() < 0.3:
        width, height = img.size
        m = random.uniform(-0.03, 0.03)
        xshift = abs(m) * width
        new_width = width + int(round(xshift))
        img = img.transform(
            (new_width, height),
            Image.Transform.AFFINE,
            (1, m, -xshift if m > 0 else 0, 0, 1, 0),
            Image.Resampling.BICUBIC,
            fillcolor="white"
        )
        img = img.resize((width, height), Image.Resampling.LANCZOS)

    # 3. Gaussian blur (hafif)
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))

    # 4. Rastgele gürültü (salt & pepper benzeri)
    if random.random() < 0.2:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 8), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # 5. Parlaklık / Kontrast
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.85, 1.15))

    return img

def create_dataset():

    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    """TXT'den kelimeleri oku, her fontla render et."""
    os.makedirs(DATASET_DIR, exist_ok=True)

    font_files = sorted([f for f in os.listdir(FONT_DIR) if f.lower().endswith(".ttf")])
    if len(font_files) == 0:
        raise FileNotFoundError(f"'{FONT_DIR}' klasöründe .ttf font bulunamadı!")

    print(f"\nBulunan fontlar ({len(font_files)}):")
    for f in font_files:
        print(f"  - {f}")

    with open(TXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    words = split_words(text)
    print(f"\nToplam kelime: {len(words)}")

    total_images = 0
    for font_name in font_files:
        font_path = os.path.join(FONT_DIR, font_name)
        label = os.path.splitext(font_name)[0]
        label_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        count = 0
        for i, word in enumerate(words):
            for aug_idx in range(3):
                img = render_word_image(word, font_path)
                if img is None:
                    continue
                if aug_idx > 0:
                    img = augment_image(img)
                save_name = f"{i:04d}_{aug_idx}.png"
                img.save(os.path.join(label_dir, save_name))
                count += 1
        print(f"  [{label}] -> {count} görüntü")
        total_images += count

    print(f"\nToplam dataset: {total_images} görüntü")
    return font_files

# ============================ 2. PYTORCH DATASET ============================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=3, fill=255),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), fill=255),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class FontImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================ 3. MODEL (ResNet-18) ============================

def get_model(num_classes):
    """Pretrained ResNet-18; gövde dondurulmuş, sadece classifier eğitilir."""
    model = models.resnet18(pretrained=True)

    # Tüm gövdeyi dondur (overfitting'i önler)
    for param in model.parameters():
        param.requires_grad = False

    # Son FC katmanını değiştir (dropout'lu)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    return model.to(DEVICE)


# ============================ 4. EĞİTİM ============================
def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    print(f"MODEL CİHAZI: {next(model.parameters()).device}")
    print(f"İLK BATCH CİHAZI: {next(iter(train_loader))[0].device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    epochs_no_improve = 0

    # ===== EĞİTİM GEÇMİŞİ KAYDETME =====
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    # ====================================

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=True):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / train_total

        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / val_total

        # ===== GEÇMİŞE KAYDET =====
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        # ==========================

        print(
            f"Epoch {epoch + 1:02d}: TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}% | TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            epochs_no_improve = 0
            print(f"  -> Yeni en iyi model kaydedildi (ValAcc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1

        scheduler.step(val_acc)

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping tetiklendi ({PATIENCE} epoch iyileşme olmadı).")
            break

    # ===== EĞİTİM BİTİNCE GRAFİK ÇİZ =====
    print("\n" + "=" * 60)
    print("EGITIM GRAFIGI")
    print("=" * 60)
    plot_training_history(history)
    # =====================================

    print(f"\nEğitim tamamlandı. En iyi doğruluk: {best_val_acc:.2f}%")
    return best_val_acc, history


def plot_training_history(history):
    """Eğitim ve validation accuracy/loss grafiğini çizer."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Headless backend (PyCharm'da sorun çıkarmaz)
        import matplotlib.pyplot as plt

        epochs = range(1, len(history['train_acc']) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy grafiği
        ax1.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', markersize=6)
        ax1.plot(epochs, history['val_acc'], 'r-s', label='Val Accuracy', markersize=6)
        ax1.set_title('Accuracy Degisimi (Epoch Bazinda)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 105])

        # Loss grafiği
        ax2.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=6)
        ax2.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', markersize=6)
        ax2.set_title('Loss Degisimi (Epoch Bazinda)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("Grafik kaydedildi: training_history.png")
        plt.close()

        # Terminalde tablo da göster
        print("\nEpoch Bazli Sonuclar:")
        print("-" * 55)
        print(f"{'Epoch':>6} | {'Train Acc':>10} | {'Val Acc':>10} | {'Train Loss':>11} | {'Val Loss':>10}")
        print("-" * 55)
        for i in range(len(history['train_acc'])):
            print(
                f"{i + 1:>6} | {history['train_acc'][i]:>9.2f}% | {history['val_acc'][i]:>9.2f}% | {history['train_loss'][i]:>11.4f} | {history['val_loss'][i]:>10.4f}")
        print("-" * 55)

    except ImportError:
        print("[UYARI] matplotlib yuklu degil, grafik cizilemiyor.")
        print("Terminalde tablo gosteriliyor:")
        print("\nEpoch Bazli Sonuclar:")
        print("-" * 40)
        print(f"{'Epoch':>6} | {'Train Acc':>10} | {'Val Acc':>10}")
        print("-" * 40)
        for i in range(len(history['train_acc'])):
            print(f"{i + 1:>6} | {history['train_acc'][i]:>9.2f}% | {history['val_acc'][i]:>9.2f}%")
        print("-" * 40)



# ============================ 5. TEST & TAHMİN ============================

def evaluate_test(model, test_loader, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Accuracy
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    total = len(all_labels)
    acc = 100 * correct / total
    print(f"\nTest Doğruluğu: {acc:.2f}%")

    # Confusion Matrix (Yazar adlarıyla)
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (Yazar Adlariyla)")
    print("=" * 60)
    cm = confusion_matrix(all_labels, all_preds)

    writer_labels = [FONT_TO_WRITER.get(c, c[:8]) for c in classes]
    header = f"{'':>12}" + "".join([f"{w[:10]:>12}" for w in writer_labels])
    print(header)

    for i in range(len(classes)):
        row_name = writer_labels[i][:10]
        row_vals = "".join([f"{cm[i][j]:>12}" for j in range(len(classes))])
        print(f"{row_name:>12}{row_vals}")

    # Classification Report (Yazar adlarıyla)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Yazar Adlariyla)")
    print("=" * 60)
    writer_names = [FONT_TO_WRITER.get(c, c) for c in classes]
    print(classification_report(all_labels, all_preds, target_names=writer_names, digits=3))

    return acc

def predict_single(model, image_path, classes, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    pred_class = classes[predicted.item()]
    conf = confidence.item()
    return pred_class, conf


def predict_detailed(model, image_path, classes, transform):
    """Tüm sınıf olasılıklarını döndürür."""
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)[0]

    probs_np = probs.cpu().numpy()
    sorted_idx = np.argsort(probs_np)[::-1]

    pred_class = classes[sorted_idx[0]]
    pred_conf = probs_np[sorted_idx[0]]
    all_scores = [(classes[i], float(probs_np[i])) for i in sorted_idx]

    return pred_class, pred_conf, all_scores

# ============================ MAIN ============================

if __name__ == "__main__":
    # 1) Dataset oluştur
    print("=" * 60)
    print("ADIM 1: Dataset oluşturuluyor...")
    print("=" * 60)
    font_files = create_dataset()
    num_classes = len(font_files)

    # 2) PyTorch Dataset & Dataloader
    print("\n" + "=" * 60)
    print("ADIM 2: DataLoader hazırlanıyor...")
    print("=" * 60)

    full_dataset = FontImageDataset(DATASET_DIR, transform=train_transform)
    val_test_base = FontImageDataset(DATASET_DIR, transform=val_transform)

    total = len(full_dataset)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    # İndeks split
    indices = list(range(total))
    random.shuffle(indices)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(val_test_base, val_idx)
    test_set = Subset(val_test_base, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Toplam: {total} | Train: {train_size} | Val: {val_size} | Test: {test_size}")
    print(f"Sınıflar: {full_dataset.classes}")

    # 3) Model oluştur
    print("\n" + "=" * 60)
    print("ADIM 3: ResNet-18 modeli oluşturuluyor...")
    print("=" * 60)
    model = get_model(num_classes)
    print(f"Sınıf sayısı: {num_classes}")
    print(f"Eğitilebilir parametre sayısı: {sum(p.numel() for p in model.fc.parameters() if p.requires_grad):,}")

    # 4) Eğitim
    print("\n" + "=" * 60)
    print("ADIM 4: Eğitim başlıyor...")
    print("=" * 60)
    best_acc, history = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    torch.save(model, "font_model_tam.pth")
    print("Tüm model kaydedildi: font_model_tam.pth")

    # 5) Test
    print("\n" + "=" * 60)
    print("ADIM 5: Test değerlendirmesi...")
    print("=" * 60)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    evaluate_test(model, test_loader, full_dataset.classes)

    # 6) Örnek tahmin
    print("\n" + "=" * 60)
    print("ADIM 6: Detayli tahminler (Test setinden 5 ornek)...")
    print("=" * 60)

    random.seed()
    sample_indices = random.sample(range(len(test_idx)), min(5, len(test_idx)))

    for i, idx in enumerate(sample_indices, 1):
        test_img_path = test_set.dataset.samples[test_idx[idx]][0]
        true_label = test_set.dataset.samples[test_idx[idx]][1]
        true_name = full_dataset.classes[true_label]

        pred, conf, all_scores = predict_detailed(
            model, test_img_path, full_dataset.classes, val_transform
        )

        # ===== YAZAR ADI DÖNÜŞÜMÜ =====
        pred_writer = FONT_TO_WRITER.get(pred, pred)
        true_writer = FONT_TO_WRITER.get(true_name, true_name)

        # Yabancı yazar kontrolü
        if pred == YABANCI_YAZAR_FONT:
            tahmin_msg = "Bu yazar bizim yazarlarımızdan değil"
        else:
            tahmin_msg = pred_writer
        # ===== DÖNÜŞÜM BİTTİ =====

        status = "DOGRU" if pred == true_name else "YANLIS"

        print(f"\n{'=' * 60}")
        print(f"{i}. GORUNTU : {os.path.basename(test_img_path)}")
        print(f"   GERCEK  : {true_writer}")
        print(f"   TAHMIN  : {tahmin_msg}  ({status})")
        print(f"   EN YUKSEK GUVEN: {conf:.2%}")
        print(f"\n   TUM FONT GUVEN SKORLARI:")
        for rank, (font_name, score) in enumerate(all_scores, 1):
            marker = "  <-- TAHMIN EDILEN" if rank == 1 else ""
            print(f"      {rank}. {font_name:<45} : {score:>6.2%}{marker}")
