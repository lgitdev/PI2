from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Désactive la limitation de taille d'image

import os


def convert_tiff_to_jpeg(input_folder, output_folder, quality=85):
    """
    Convertit toutes les images TIFF en JPEG avec compression ajustable.

    :param input_folder: Dossier contenant les images originales (.tif)
    :param output_folder: Dossier où enregistrer les images JPEG
    :param quality: Qualité JPEG (100 = meilleure qualité, 50 = forte compression)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(".tif") or file_name.lower().endswith(".tiff"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".jpg")

            try:
                with Image.open(input_path) as img:
                    print(f"🔍 Conversion de {file_name} - Mode : {img.mode}, Taille : {img.size}")

                    # Convertir en RGB si nécessaire (JPEG ne supporte pas tous les modes)
                    if img.mode not in ["RGB", "L"]:
                        img = img.convert("RGB")

                    # Sauvegarde en JPEG avec compression
                    img.save(output_path, format="JPEG", quality=quality, optimize=True)
                    print(f"✅ {file_name} converti en JPEG ({quality}% qualité)")

            except Exception as e:
                print(f"❌ Erreur avec {file_name} : {e}")

# 🔧 Configuration des dossiers (à modifier selon votre besoin)
input_folder = r"C:\\Users\\gindr\\Documents\\2024-2025\\ESILV\\Cours\\S8\\PI2\\PI2\\1980"
output_folder = r"C:\\Users\\gindr\\Documents\\2024-2025\\ESILV\\Cours\\S8\\PI2\\PI2\\to_match\\1980"

# 🏃‍♂️ Exécuter la compression
convert_tiff_to_jpeg(input_folder, output_folder, quality=50)
