import cv2 as cv
import numpy as np
import matching as geo

"""
Ce script calcul le taux de suppression des corespondances entre plusieurs couples de photos pris dans différentes missions 
sur deux années.
"""


# Dictionary to store images with their corresponding filenames
image_dict = {
    "img2443": cv.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2443.tif"),
    "img2444": cv.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2444.tif"),
    "img2591": cv.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2591.tif"),
    "img2590": cv.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2590.tif"),
    "img2589": cv.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2589.tif"),
    "img4375": cv.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4375.tif"),
    "img4374": cv.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4374.tif"),
}

# Convert dictionary to two separate lists
IMG = list(image_dict.values())  # Images
IMG_NAMES = list(image_dict.keys())  # Corresponding names

# Taille d'image pour visualisation du procédé
for i in range(len(IMG)) :
    scale_percent = 5  # 5%, peut être adapté
    width = int(IMG[i].shape[1] * scale_percent / 100)
    height = int(IMG[i].shape[0] * scale_percent / 100)
    dim = (width, height)
    IMG[i] = cv.resize(IMG[i], dim, interpolation=cv.INTER_AREA)


angles = np.zeros((len(IMG),len(IMG)))

for i in range(len(IMG)) :
    for j in range(i+1, len(IMG)) :
        kpt1, kpt2, matches = geo.init_matching_orb(IMG[i], IMG[j])
        matches = geo.filtre_distance(matches)
        lines = cv.drawMatches(IMG[i], kpt1, IMG[j], kpt2, matches, None)

        # RANSAC
        H, mask = geo.ransac(kpt1, kpt2, matches)
        mask = mask.ravel().tolist()
        filtered_lines = cv.drawMatches(IMG[i], kpt1, IMG[j], kpt2, matches, None, matchColor=(0,255,0), matchesMask=mask, flags=2)
        angles[i][j]=geo.extract_rotation_angle(H)
        angles[j][i]=geo.extract_rotation_angle(H)


        #VISU
        cv.imwrite(f'Matches entre {IMG_NAMES[i]} et {IMG_NAMES[j]}.jpg', filtered_lines)

        cv.waitKey(0)
        cv.destroyAllWindows()

geo.matrice_angle(angles)

