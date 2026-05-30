# prepare.py
alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#&'()*+,-./:;? "
with open('alphabet.txt', 'w', encoding='utf-8') as f:
    f.write(alphabet)
print("Alfabe (alphabet.txt) hazır. İki eğitim için de bu kullanılacak.")