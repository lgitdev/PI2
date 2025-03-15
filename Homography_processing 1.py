import numpy as np
import cv2
import matching as match
import os

# Définition du chemin des images
image_folder = "C:\\Users\\gindr\\Documents\\2024-2025\\ESILV\\Cours\\S8\\PI2\\PI2\\photos"

# Liste des images à charger

image_filenames = [
    "2007_P08000262_2071.tif", "2007_P08000262_2176.tif", "2007_P08000262_2177.tif",
    "2007_P08000262_2178.tif", "2007_P08000262_2330.tif", "2007_P08000262_2442.tif",
    "2007_P08000262_2443.tif", "2007_P08000262_2444.tif", "2007_P08000262_2590.tif",
    "2007_P08000262_4374.tif", "2007_P08000262_4375.tif", "2007_P08000262_4376.tif",
    "2007_P08000262_4377.tif", "2007_P08000262_4378.tif", "2007_P08000262_4379.tif",
    "2007_P08000262_4516.tif", "2007_P08000262_4515.tif", "2007_P08000262_4643.tif"
]
'''
image_filenames = [
    "2013_P14000772_1729.tif", "2013_P14000772_1885.tif", "2013_P14000772_1837.tif",
    "2013_P14000772_2004.tif", "2013_P14000772_2047.tif", "2013_P14000772_2169.tif",
    "2013_P14000772_2208.tif", "2013_P14000772_2325.tif"
]'''

# Chargement des images
IMG = [cv2.imread(os.path.join(image_folder, filename)) for filename in image_filenames]

# Taille d'image pour visualisation du procédé
for i in range(len(IMG)) :
    scale_percent = 5  # 5%, peut être adapté
    width = int(IMG[i].shape[1] * scale_percent / 100)
    height = int(IMG[i].shape[0] * scale_percent / 100)
    dim = (width, height)
    IMG[i] = cv2.resize(IMG[i], dim, interpolation=cv2.INTER_AREA)

    
#Storing image sizes
h,w = [], []
for i in range(len(IMG)):
    h.append(IMG[i].shape[0])
    w.append(IMG[i].shape[1])

# Create a canvas big enough to fit both images
canvas = np.zeros((h[0]+200, w[0]+200, 3) , dtype=np.uint8)

# Place the first image at (0,0)
canvas[200:h[0]+200, 200:w[0]+200] = IMG[0]
x_offset=200
y_offset=200


#Matching
for i in range(len(IMG)-1):
    print(i+1)
    kpt1, kpt2, matches = match.init_matching_orb(IMG[i], IMG[i+1])
    matches = match.filtre_distance(matches)

    # RANSAC
    H, mask = match.ransac(kpt1, kpt2, matches)
    mask = mask.ravel().tolist()
    filtered_lines = cv2.drawMatches(IMG[i], kpt1, IMG[i+1], kpt2, matches, None, matchColor=(0,255,0), matchesMask=mask, flags=2)
        
    #cv2.imwrite(f'Homography {i} and {i+1}.jpg', filtered_lines)

    # Extract translation from homography
    tx, ty = -H[0, 2], -H[1, 2]
    print(f"Translation: ({tx}, {ty})")

    #cv2.imwrite("Canvas.jpg", canvas)
    #Increasing the canvas to fit the new image
    new_canvas = np.zeros((canvas.shape[0]+ abs(int(ty)), canvas.shape[1] + abs(int(tx)), 3), dtype=np.uint8)
    new_canvas[:canvas.shape[0], :canvas.shape[1]] = canvas
    canvas = new_canvas
    #cv2.imwrite("New Canvas.jpg", canvas)

    # Place the second image at its computed offset
    x_offset += int(tx)
    y_offset += int(ty)
    print(f"offset x = {x_offset} and offset y = {y_offset}")
    canvas[ y_offset:y_offset + h[i+1] ,x_offset:x_offset + w[i+1]] = IMG[i+1]

# Show result
cv2.imwrite("Aligned Placement.jpg", canvas)

cv2.waitKey(0)
cv2.destroyAllWindows()
