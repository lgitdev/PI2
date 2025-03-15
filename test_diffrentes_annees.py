import cv2 as cv
import numpy as np
import matching as geo
import os
import re

"""
Ce script effectue un matching entre les images proches géographiquement dans la base de données, et vérifie que le matching est cohérent
en utilisant des KPIs tels que la moyenne angulaire ou le parallélisme du matching.
Il n'affiche QUE les lignes de matching correspondant aux droites parallèles.
"""

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

def find_parallel_groups(angles, tolerance=1):
    """ Trouve les groupes de lignes ayant des angles similaires """
    angle_groups = {}
    
    for angle in angles:
        found_group = False
        for key in angle_groups.keys():
            if abs(angle - key) < tolerance:
                angle_groups[key].append(angle)
                found_group = True
                break
        if not found_group:
            angle_groups[angle] = [angle]

    # Trouver le groupe avec le plus de lignes parallèles
    best_group = max(angle_groups.values(), key=len, default=[])
    return best_group

def filter_parallel_matches(kpt1, kpt2, matches, tolerance=1):
    angles = []

    for match in matches:
        pt1 = np.array(kpt1[match.queryIdx].pt)
        pt2 = np.array(kpt2[match.trainIdx].pt)
        delta = pt2 - pt1
        angle = np.arctan2(delta[1], delta[0]) * 180 / np.pi
        angles.append((angle, match))

    best_group = find_parallel_groups([a[0] for a in angles], tolerance)
    
    if len(best_group) < 5:
        return []  # Si moins de 5 lignes, on n'affiche rien

    # Filtrer les correspondances qui appartiennent au groupe parallèle dominant
    filtered_matches = [match for angle, match in angles if angle in best_group]
    return filtered_matches

angles = np.zeros((len(IMG), len(IMG)))
SEUIL_KPI = 30.0  

for i in range(len(IMG)):
    for j in range(i + 1, min(i + 4, len(IMG))):
        print(f"Matching entre {IMG_NAMES[i]} et {IMG_NAMES[j]}...")

        kpt1, kpt2, matches = geo.init_matching_orb(IMG[i], IMG[j])
        matches = geo.filtre_distance(matches)

        if len(matches) < 4:
            print(f"❌ Skipping {IMG_NAMES[i]} - {IMG_NAMES[j]} (pas assez de correspondances : {len(matches)})")
            continue

        # Filtrer pour ne garder que les matches correspondant aux lignes parallèles
        matches = filter_parallel_matches(kpt1, kpt2, matches)

        if len(matches) < 5:
            print(f"❌ Skipping {IMG_NAMES[i]} - {IMG_NAMES[j]} (pas assez de lignes parallèles)")
            continue

        H, mask = geo.ransac(kpt1, kpt2, matches)
        mask = mask.ravel().tolist()
        filtered_lines = cv.drawMatches(
            IMG[i], kpt1, IMG[j], kpt2, matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=mask,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        angles[i][j] = geo.extract_rotation_angle(H)
        angles[j][i] = geo.extract_rotation_angle(H)

        if abs(angles[i][j]) > SEUIL_KPI:
            print(f"❌ Skipping {IMG_NAMES[i]} - {IMG_NAMES[j]} (écart angulaire {angles[i][j]:.2f}° > seuil {SEUIL_KPI}°)")
            continue

        cv.imwrite(f'Matches_{IMG_NAMES[i]}_et_{IMG_NAMES[j]}.jpg', filtered_lines)
        print(f"✅ Matching validé pour {IMG_NAMES[i]} et {IMG_NAMES[j]}.")

geo.matrice_angle(angles)
print("✅ Traitement terminé.")
