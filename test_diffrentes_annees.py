import cv2 as cv
import numpy as np
import matching as geo
import os
import re

"""
Ce script effectue un matching entre les images proches géographiquement dans la base de données, et vérifie que le matching est cohérent
en utilisant des KPIs tels que la moyenne angulaire ou le parallélisme du matching
"""

image_folder = "C:\\Users\\gindr\\Documents\\2024-2025\\ESILV\\Cours\\S8\\PI2\\PI2\\photos\\2007"
image_filenames = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.tif'))]

# Trier les images par ordre croissant -> on ne garde que les voisins les plus proches
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

image_filenames.sort(key=extract_number)

image_dict = {filename: cv.imread(os.path.join(image_folder, filename)) for filename in image_filenames}

IMG = list(image_dict.values())  # Liste des images chargées
IMG_NAMES = list(image_dict.keys())  # Liste des noms d'images

# Vérifier si toutes les images sont bien chargées
for name, img in image_dict.items():
    if img is None:
        print(f"Erreur : L'image {name} n'a pas pu être chargée.")

if not IMG:
    print("Aucune image trouvée dans le dossier spécifié.")
    exit()

# Fonction pour vérifier le parallélisme des lignes de matching
def check_parallel_matches(kpt1, kpt2, matches, tolerance=5, min_parallel=5):
    angles = []

    for match in matches:
        pt1 = np.array(kpt1[match.queryIdx].pt)  # Point dans la première image
        pt2 = np.array(kpt2[match.trainIdx].pt)  # Point correspondant dans la deuxième image

        # Calcul de l'angle de la ligne de matching
        delta = pt2 - pt1
        angle = np.arctan2(delta[1], delta[0]) * 180 / np.pi  # Conversion en degrés
        angles.append(angle)

    if len(angles) < min_parallel:
        return False  # Pas assez de lignes pour vérifier

    # Vérification du nombre de lignes quasi-parallèles
    angles = np.array(angles)
    for ref_angle in angles:
        count_parallel = np.sum(np.abs(angles - ref_angle) < tolerance)  # Vérifie si l'écart est < tolérance
        if count_parallel >= min_parallel:
            return True  # Au moins 5 lignes parallèles trouvées

    return False  # Pas assez de lignes parallèles

# Initialisation de la matrice des angles
angles = np.zeros((len(IMG), len(IMG)))

# Seuil KPI (exemple : on garde uniquement les matchings où l'angle moyen est inférieur à 10°)
SEUIL_KPI = 10.0  

# Matching entre chaque image et ses 3 voisins supérieurs
for i in range(len(IMG)):
    for j in range(i + 1, min(i + 4, len(IMG))):  # Maximum 3 voisins supérieurs
        print(f"Matching entre {IMG_NAMES[i]} et {IMG_NAMES[j]}...")

        kpt1, kpt2, matches = geo.init_matching_orb(IMG[i], IMG[j])
        matches = geo.filtre_distance(matches)

        # Vérifier qu'on a au moins 4 correspondances valides
        if len(matches) < 4:
            print(f"❌ Skipping {IMG_NAMES[i]} - {IMG_NAMES[j]} (pas assez de correspondances : {len(matches)})")
            continue  # Ignore cette paire d'images

        # Vérifier la présence de 5 lignes parallèles
        if not check_parallel_matches(kpt1, kpt2, matches):
            print(f"❌ Skipping {IMG_NAMES[i]} - {IMG_NAMES[j]} (pas assez de lignes parallèles)")
            continue  # Ignore cette paire d'images

        # RANSAC pour l'homographie
        H, mask = geo.ransac(kpt1, kpt2, matches)
        mask = mask.ravel().tolist()
        filtered_lines = cv.drawMatches(IMG[i], kpt1, IMG[j], kpt2, matches, None, matchColor=(0,255,0), matchesMask=mask, flags=2)

        # Calcul de l'angle de rotation
        angles[i][j] = geo.extract_rotation_angle(H)
        angles[j][i] = geo.extract_rotation_angle(H)

        # Vérification du KPI (écart angulaire)
        if abs(angles[i][j]) > SEUIL_KPI:
            print(f"❌ Skipping {IMG_NAMES[i]} - {IMG_NAMES[j]} (écart angulaire {angles[i][j]:.2f}° > seuil {SEUIL_KPI}°)")
            continue  # Ignore cette paire si l'angle dépasse le seuil

        # Sauvegarde des images de matching valides
        cv.imwrite(f'Matches_{IMG_NAMES[i]}_et_{IMG_NAMES[j]}.jpg', filtered_lines)
        print(f"✅ Matching validé pour {IMG_NAMES[i]} et {IMG_NAMES[j]}.")

# Affichage de la matrice des angles
geo.matrice_angle(angles)

print("✅ Traitement terminé.")
