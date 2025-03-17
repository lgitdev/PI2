import cv2 as cv
import numpy as np
import matching as geo
import os
import re

# Chargement des images
image_folder = "C:\\Users\\gindr\\Documents\\2024-2025\\ESILV\\Cours\\S8\\PI2\\PI2\\to_match\\1980"
image_filenames = [f for f in os.listdir(image_folder) if f.endswith(('.jpg'))]

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

image_filenames.sort(key=extract_number)
image_dict = {filename: cv.imread(os.path.join(image_folder, filename)) for filename in image_filenames}
IMG_NAMES = list(image_dict.keys())

def crop_black_borders(img, threshold=20, dark_ratio=0.2):
    """
    Supprime les bordures noires d'une image si plus de `dark_ratio` (20% par défaut) d'une ligne ou colonne est trop sombre.
    """
    if img is None:
        raise ValueError("L'image fournie est invalide ou introuvable.")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Détection des lignes et colonnes sombres
    row_dark_ratio = np.sum(gray < threshold, axis=1) / width
    col_dark_ratio = np.sum(gray < threshold, axis=0) / height

    # Trouver les indices des premières et dernières lignes/colonnes non sombres
    top = np.argmax(row_dark_ratio < dark_ratio)
    bottom = height - np.argmax(row_dark_ratio[::-1] < dark_ratio)
    left = np.argmax(col_dark_ratio < dark_ratio)
    right = width - np.argmax(col_dark_ratio[::-1] < dark_ratio)

    # Vérification pour éviter un recadrage vide
    if bottom <= top or right <= left:
        print("⚠️ Recadrage impossible, l'image est probablement entièrement noire.")
        return img  # Retourner l'image d'origine si le recadrage échoue

    # Recadrer l'image
    cropped_image = img[top:bottom, left:right]
    return cropped_image

# Redimensionnement pour le traitement (10%) et sauvegarde des versions originales
scale_percent_processing = 10
resized_images = {}

for name in IMG_NAMES:
    img = image_dict[name]
    img = crop_black_borders(img)
    width = int(img.shape[1] * scale_percent_processing / 100)
    height = int(img.shape[0] * scale_percent_processing / 100)
    resized_images[name] = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

def find_parallel_groups(angles, tolerance=1):
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

    best_group = max(angle_groups.values(), key=len, default=[])
    return best_group

def filter_parallel_matches(kpt1, kpt2, matches, tolerance=0.1):
    angles = []

    for match in matches:
        pt1 = np.array(kpt1[match.queryIdx].pt)
        pt2 = np.array(kpt2[match.trainIdx].pt)
        delta = pt2 - pt1
        angle = np.arctan2(delta[1], delta[0]) * 180 / np.pi
        angles.append((angle, match))

    best_group = find_parallel_groups([a[0] for a in angles], tolerance)
    
    if len(best_group) < 5:
        return []

    filtered_matches = [match for angle, match in angles if angle in best_group]
    return filtered_matches

# Matching et stitching généralisé
stitched_image = None
stitched_name = None
stitching_index = 0  # Variable pour incrémenter les fichiers stitched

matchs = set()
while True:
    found_match = False
    keys_list = list(image_dict.keys())  # Liste dynamique

    for i in range(len(keys_list)):
        for j in range(i + 1, len(keys_list)):
            img1_name = keys_list[i]
            img2_name = keys_list[j]

            if (img1_name, img2_name) in matchs or (img2_name, img1_name) in matchs:
                continue
            matchs.add((img1_name, img2_name))

            print(f"🔍 Tentative de matching entre {img1_name} et {img2_name}...")

            img1 = resized_images[img1_name]  # Version réduite pour traitement
            img2 = resized_images[img2_name]  # Version réduite pour traitement

            kpt1, kpt2, matches = geo.init_matching_orb(img1, img2)
            matches = geo.filtre_distance(matches)
            if len(matches) < 4:
                continue

            matches = filter_parallel_matches(kpt1, kpt2, matches)
            if len(matches) < 5:
                continue

            H, mask = geo.ransac(kpt1, kpt2, matches)
            if H is None:
                print(f"⚠️ Homographie impossible entre {img1_name} et {img2_name}")
                continue  # Recommence avec les images restantes

            mask = mask.ravel().tolist()

            angles = geo.extract_rotation_angle(H)
            SEUIL_KPI = 30.0

            if abs(angles) > SEUIL_KPI:
                continue

            print(f"✅ Match validé entre {img1_name} et {img2_name} !")

            
            # VISUALISATION DU MATCHING
            matching_visualization = cv.drawMatches(
                img1, kpt1, img2, kpt2, matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                matchesMask=mask,
                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            matching_filename = f"matching_{stitching_index+1}.jpg"
            cv.imwrite(matching_filename, matching_visualization)
            print(f"📸 Visualisation des correspondances sauvegardée sous {matching_filename}")

            # Stitching avec les images originales (100 %)
            img1_full = image_dict[img1_name]
            img2_full = image_dict[img2_name]

            x1, y1 = map(int, kpt1[matches[0].queryIdx].pt)
            x2, y2 = map(int, kpt2[matches[0].trainIdx].pt)
            dx, dy = x1 - x2, y1 - y2

            h1, w1 = img1_full.shape[:2]
            h2, w2 = img2_full.shape[:2]
            canvas_width = max(w1 + abs(dx), w2 + abs(dx))
            canvas_height = max(h1 + abs(dy), h2 + abs(dy))
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            x_start = -dx if dx < 0 else 0
            y_start = -dy if dy < 0 else 0

            canvas[y_start:y_start + h1, x_start:x_start + w1] = img1_full
            x_offset = -dx if dx < 0 else 0  
            y_offset = -dy if dy < 0 else 0  

            M = np.float32([[1, 0, x_offset + dx], [0, 1, y_offset + dy]])
            transformed_img = cv.warpAffine(img2_full, M, (canvas_width, canvas_height))

            mask = (transformed_img > 0).astype(np.uint8)
            canvas = canvas * (1 - mask) + transformed_img * mask

            stitched_cropped = crop_black_borders(canvas)
            

            stitching_index += 1
            
            stitched_name = f"stitching_{stitching_index}.jpg"
            stitched_image = stitched_cropped

            width_resized = int(stitched_cropped.shape[1] * scale_percent_processing / 100)
            height_resized = int(stitched_cropped.shape[0] * scale_percent_processing / 100)
            resized_images[stitched_name] = cv.resize(stitched_cropped, (width_resized, height_resized), interpolation=cv.INTER_AREA)


            del image_dict[img1_name]
            del image_dict[img2_name]

            image_dict = {**image_dict, stitched_name: stitched_cropped}

            found_match = True
            break

        if found_match:
            break

    if not found_match:
        print("✅ Stitching terminé, plus aucune correspondance trouvée.")
        break

if stitched_image is not None:
    cv.imshow("Final Stitched Image", stitched_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite(stitched_name, stitched_image)
    print(f"✅ Image finale sauvegardée sous {stitched_name}")
