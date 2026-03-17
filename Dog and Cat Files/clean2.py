import tensorflow as tf
import os
import glob

# BURAYI GÜNCELLE: Veri setinin olduğu ana klasör yolu
# (İçinde kedi ve köpek klasörlerinin olduğu yer)
dataset_path = "PetImages"

print("TensorFlow ile derin tarama başlıyor...")
print("Bu işlem biraz sürebilir, lütfen bekleyin...")

# Tüm alt klasörlerdeki resimleri bul (jpg, jpeg, png, bmp)
files = glob.glob(os.path.join(dataset_path, "*", "*"))

deleted_count = 0

for file_path in files:
    # Sadece resim dosyalarına bak
    if not file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue

    try:
        # 1. Dosyayı ham byte olarak oku
        file_bytes = tf.io.read_file(file_path)
        
        # 2. TensorFlow'un decode etmesini iste
        # Hata buradaysa, 'try' bloğu 'except'e düşecektir.
        img = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
        
        # 3. İsteğe bağlı: Boyut kontrolü (0 byte vb.)
        if img.shape is None:
            raise Exception("Resim boyutu okunamadı.")
            
    except Exception as e:
        print(f"BOZUK DOSYA BULUNDU VE SİLİNİYOR: {file_path}")
        print(f"Hata Sebebi: {e}")
        try:
            os.remove(file_path)
            deleted_count += 1
        except:
            print("Dosya silinemedi (Kullanımda olabilir).")

print(f"\nTarama bitti! Toplam {deleted_count} adet bozuk dosya silindi.")