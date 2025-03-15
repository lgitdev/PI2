import numpy as np
import cv2
import matching as match

img02 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2071.tif")
img01 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2176.tif")
img0 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2177.tif")
img1 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2178.tif")
img2 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2330.tif") 
img3 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2442.tif") 
img4 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2443.tif")
img5 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2444.tif")
img6 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_2590.tif")
img65 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4374.tif")
img7 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4375.tif")
img8 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4376.tif")
img9 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4377.tif")
img10 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4378.tif")
img11 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4379.tif")
img12= cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4516.tif")
img13= cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4515.tif")
img14= cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2007/2007_P08000262_4643.tif")

IMG = [img02,img01,img0,img1,img2,img3,img4,img6,img65,img7,img8,img9,img10,img11,img12,img13,img14]

"""img1 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2013/2013_P14000772_1729.tiff")
img2 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2013/2013_P14000772_1837.tiff") 
img3 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2013/2013_P14000772_1885.tiff") 
img4 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2013/2013_P14000772_2004.tiff")
img5 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2013/2013_P14000772_2047.tiff")
img6 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2013/2013_P14000772_2169.tiff")
img7 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2013/2013_P14000772_2208.tiff")
img8 = cv2.imread("C:/Users/exill/Documents/Erdre/Git Clone 2/Erdre/2013/2013_P14000772_2325.tiff")"""

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
