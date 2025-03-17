import cv2 as cv
import numpy as np
import matching as geo
import os
import re

# Dossier contenant les images
image_folder = "C:\\Users\\gindr\\Documents\\2024-2025\\ESILV\\Cours\\S8\\PI2\\PI2\\photos\\2007"
image_filenames = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.tif'))]

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

image_filenames.sort(key=extract_number)
image_dict = {filename: cv.imread(os.path.join(image_folder, filename)) for filename in image_filenames}
IMG = list(image_dict.values())
IMG_NAMES = list(image_dict.keys())

for name, img in image_dict.items():
    if img is None:
        print(f"Erreur : L'image {name} n'a pas pu être chargée.")

if not IMG:
    print("Aucune image trouvée dans le dossier spécifié.")
    exit()

scale_percent = 10
for i in range(len(IMG)):
    width = int(IMG[i].shape[1] * scale_percent / 100)
    height = int(IMG[i].shape[0] * scale_percent / 100)
    IMG[i] = cv.resize(IMG[i], (width, height), interpolation=cv.INTER_AREA)

SEUIL_KPI = 30.0  
base_image = IMG[0]  # Première image comme référence

# Dimensions maximales estimées (on agrandit pour être sûr)
max_height, max_width = base_image.shape[:2]
canvas = np.zeros((max_height * 3, max_width * 3, 3), dtype=np.uint8)

# Placer la première image au centre du canevas
h, w = base_image.shape[:2]
center_x, center_y = max_width, max_height
canvas[center_y:center_y + h, center_x:center_x + w] = base_image

# Définir la transformation initiale
base_transform = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])

for i in range(len(IMG) - 1):
    print(f"Stitching entre {IMG_NAMES[i]} et {IMG_NAMES[i+1]}...")

    kpt1, kpt2, matches = geo.init_matching_orb(IMG[i], IMG[i+1])
    matches = geo.filtre_distance(matches)

    if len(matches) < 4:
        print(f"❌ Skipping {IMG_NAMES[i]} - {IMG_NAMES[i+1]} (pas assez de correspondances : {len(matches)})")
        continue

    H, mask = geo.ransac(kpt1, kpt2, matches)
    if H is None:
        print(f"❌ Homographie non trouvée entre {IMG_NAMES[i]} et {IMG_NAMES[i+1]}")
        continue

    # Appliquer la transformation à l'image suivante
    full_transform = np.dot(base_transform, H)
    height2, width2 = IMG[i+1].shape[:2]
    transformed_img = cv.warpPerspective(IMG[i+1], full_transform, (canvas.shape[1], canvas.shape[0]))

    # Fusionner avec le canevas existant
    mask = (transformed_img > 0).astype(np.uint8)  # Masque des pixels valides
    canvas = cv.addWeighted(canvas, 1, transformed_img, 1, 0) * mask + canvas * (1 - mask)

    # Mise à jour de la transformation de base pour la prochaine image
    base_transform = full_transform.copy()

cv.imwrite("stitched_output.jpg", canvas)
print("✅ Stitching terminé. Image sauvegardée sous 'stitched_output.jpg'.")
