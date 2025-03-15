import cv2 as cv # OpenCV, bibliothèque spécialisée dans le traitement des images
import numpy as np
import matplotlib.pyplot as plt



def init_matching_orb(img1, img2):
    """
    Prend en argument deux images cv2, trouvent des points particuliers dans chacune d'entre elles puis essaie de coupler les paires
    de points clés qui se ressemblent le plus.
    Renvoie les points clés des deux images et les pairs couplées.
    """
    # Détecter les points clés et les descripteurs avec ORB
    orb = cv.ORB_create()
    kpt1, desc1 = orb.detectAndCompute(img1, None)
    kpt2, desc2 = orb.detectAndCompute(img2, None)

    # Matcher les points clés avec BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    #matches = filtre_distance(matches) # A choisir: applique-t-on le filtre ici ou se garde-t-on la possibilité ne pas le faire ?
    return kpt1, kpt2, matches

def init_matching_sift(img1, img2):
    """
    Prend en argument deux images cv2, trouvent des points particuliers dans chacune d'entre elles puis essaie de coupler les paires
    de points clés qui se ressemblent le plus.
    Renvoie les points clés des deux images et les pairs couplées.
    """
    # Détecter les points clés et les descripteurs avec SIFT
    sift = cv.SIFT_create()
    kpt1, desc1 = sift.detectAndCompute(img1, None)
    kpt2, desc2 = sift.detectAndCompute(img2, None)

    # Matcher les points clés avec BFMatcher
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kpt1, kpt2, matches

def filtre_distance(matches):
    """
    Prend en entrée des matchs BFMatcher et renvoie seulement ceux qui passe le fiiltre.
    Ici on définie une distance maximal pour accepter des couples de points clés comme pertinents dans BFMatcher.
    """
    # Exemple pour calculer la distance moyenne et l'écart type
    distances = [m.distance for m in matches]
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    # Définir un seuil
    threshold = mean_distance + std_distance
    # Filtrer les matches
    return [m for m in matches if m.distance < threshold]

def ransac(kpt1, kpt2, matches, seuil_ransac = 5.0):
    """
    Prend en entrée les points clés et les couples.
    Essaie de trouver une homographie (ie une fonction qui conserve les angles, donc translation/rotation/changement d'échelle) 
    qui convienne à un maximum de couples et supprime les autres.
    Renvoie l'homographie sous forme d'une matrice et le masque.
    """
    x = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    y = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(x, y, cv.RANSAC, seuil_ransac)
    #mask = mask.ravel().tolist()
    return H, mask

def taux_suppression(matches, mask_list): # Attention ici le masque est une liste et pas un objet cv2
    """
    Prend en entrée la lsite des matchs de base, et ce qu'il en reste dans le masque après le filtre et ransac (en entrée).
    Renvoie le pourcentage de matchs originaux qui sont conservés à la fin
    """
    return 100*(sum(mask_list)/len(matches))

import numpy as np

def extract_rotation_angle(H):
    """
    Extracts the rotation angle (in degrees) from a homography matrix.
    """
    if H is None:
        return None  # No valid homography

    r11, r12 = H[0, 0], H[0, 1]
    r21, r22 = H[1, 0], H[1, 1]

    theta = np.arctan2(r21, r11)  # Compute rotation in radians
    angle = np.degrees(theta)  # Convert to degrees

    return angle

def matrice_angle(list_angle, noms_images=None):
    """
    Affiche une matrice de taux de suppression dans une fenêtre graphique en utilisant une heatmap.
    :param taux_suppression: liste de listes de floats représentant les taux de suppression pour chaque couple d'images.
    """
    if noms_images == None :
        noms_images = range(len(list_angle))

    # Conversion de la matrice en array NumPy pour faciliter la gestion
    array_angle = np.array(list_angle)

    # Création de la figure et de la heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(array_angle, cmap='coolwarm', interpolation='nearest')
    
    # Ajout des annotations de valeur dans chaque case
    for i in range(array_angle.shape[0]):
        for j in range(array_angle.shape[1]):
            plt.text(j, i, f"{array_angle[i, j]:.2f}", ha='center', va='center', color='black')
    
    # Ajout des noms des images comme étiquettes d'axes
    plt.xticks(ticks=np.arange(len(noms_images)), labels=noms_images, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(noms_images)), labels=noms_images)

    # Ajout d'une barre de couleur
    plt.colorbar(label="angles (°)")
    
    # Labels des axes
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")
    plt.title("Matrice des angles")
    
    # Affichage
    plt.show()

def rescale(img, scale_percent = 5):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)
