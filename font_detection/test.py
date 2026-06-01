"""
FONT TAHMİN BÖLÜMÜ - Kaydedilmiş Modeli Kullanarak Tahmin Etme (Geliştirilmiş Kırpma Sürümü)
========================================================================================
"""
import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
# ============================ AYARLAR ============================
MODEL_PATH = "fontt_model_tam.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = [
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
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
def crop_and_binarize_robust(img, padding=15):
    """
    Fotoğrafı çekilmiş kağıt üzerindeki yazıları temizlemek için dinamik eşikleme (binarizasyon) yapar.
    Kağıt üzerindeki gölgeleri ve dokuyu yok ederek saf siyah-beyaz (tıpkı eğitim seti gibi) hale getirir.
    """
    # 1. Gri tona çevir
    gray = img.convert("L")
    arr = np.array(gray)

    # 2. Resimdeki en koyu (yazı) ve en parlak (kağıt) piksel değerlerini bul
    min_val = float(arr.min())
    max_val = float(arr.max())

    # Kontrast çok düşükse (örneğin tamamen boş resimse) işlem yapma
    if max_val - min_val < 20:
        return img

    # 3. Dinamik eşik (threshold) belirle (%65 kuralı)
    # Kağıt rengi (max) ile yazı rengi (min) arasındaki mesafeye göre eşik değeri
    threshold = min_val + (max_val - min_val) * 0.65

    # 4. Resmi ikiliye (Binarize) dönüştür: Eşik altı 0 (siyah yazı), üstü 255 (saf beyaz arka plan)
    binary_arr = np.where(arr < threshold, 0, 255).astype(np.uint8)

    # 5. Yazının sınır kutusunu (bounding box) bul (0 olan pikseller)
    y_indices, x_indices = np.where(binary_arr == 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return img

    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # 6. Kenarlık payı ekle
    w, h = img.size
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    # 7. Siyah-beyaz resmi oluştur ve kırp (Eğitim setindeki gibi saf arka plan elde ederiz)
    binary_img = Image.fromarray(binary_arr).convert("RGB")
    cropped = binary_img.crop((x_min, y_min, x_max, y_max))

    # Hata ayıklama için isterseniz kırpılmış resmi kaydedip bakabilirsiniz:
    # cropped.save("debug_cropped.png")

    return cropped
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"[HATA] Görsel bulunamadı: {image_path}")
        return
    # 1. Modeli yükle
    print(f"Model yükleniyor: {MODEL_PATH} ({DEVICE})")
    try:
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        model.eval()
    except Exception as e:
        print(f"[HATA] Model yüklenemedi! Hata: {e}")
        return
    # 2. Resmi yükle
    image = Image.open(image_path).convert("RGB")

    # DİNAMİK BİNARİZASYON VE KIRPMA
    cleaned_image = crop_and_binarize_robust(image)

    # Transform uygula
    tensor_img = predict_transform(cleaned_image).unsqueeze(0).to(DEVICE)
    # 3. Tahmin yap
    with torch.no_grad():
        output = model(tensor_img)
        probs = torch.softmax(output, dim=1)[0]

    probs_np = probs.cpu().numpy()

    # Sonuçları yazdır
    print("\n" + "=" * 50)
    print(f" GÖRSEL: {os.path.basename(image_path)}")
    print("=" * 50)

    for i, score in enumerate(probs_np):
        font_name = CLASSES[i]
        writer_name = FONT_TO_WRITER.get(font_name, font_name)
        print(f"{writer_name:<15} ({font_name[:15]}...): %{score*100:.2f}")

    best_idx = probs_np.argmax()
    best_font = CLASSES[best_idx]
    best_writer = FONT_TO_WRITER.get(best_font, best_font)
    print("-" * 50)
    print(f"EN GÜÇLÜ TAHMİN: {best_writer} (%{probs_np[best_idx]*100:.2f})")
    print("=" * 50)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "yeni_görsel.png"
        print(f"[BİLGİ] Resim yolu belirtilmedi. Varsayılan olarak '{img_path}' test edilecek.")
        print("Kullanım: python predict.py <resim_yolu>")
        print("-" * 50)

    predict_image(img_path)
