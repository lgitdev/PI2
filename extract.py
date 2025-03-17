import cv2 as cv
import os

# Dossier contenant les images
image_folder = "C:\\Users\\gindr\\Documents\\2024-2025\\ESILV\\Cours\\S8\\PI2\\PI2\\to_match\\2007"

# Extensions d'images à traiter
valid_extensions = ('.jpg', '.png', '.tif', '.jpeg')

# Liste des fichiers images
image_filenames = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]

# Compression des images
for filename in image_filenames:
    img_path = os.path.join(image_folder, filename)

    # Charger l'image
    img = cv.imread(img_path)

    if img is None:
        print(f"❌ Impossible de lire {filename}, fichier peut-être corrompu.")
        continue

    # Réduire la taille par 10
    new_width = img.shape[1] // 10
    new_height = img.shape[0] // 10

    if new_width == 0 or new_height == 0:
        print(f"⚠️ Image {filename} trop petite pour être réduite davantage.")
        continue

    resized_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)

    # Sauvegarder en remplaçant l'original
    cv.imwrite(img_path, resized_img)
    print(f"✅ Image compressée et remplacée : {filename}")

print("✅ Compression terminée pour toutes les images.")
