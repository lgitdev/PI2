import cv2 as cv
import pandas as pd
import numpy as np
import os
import matching

# BIBLIOTHEQUE DU PROJET PI2 DE LA FEDERATION DES AMIS DE L'ERDRE
# DEFINITION DES CLASSES IMAGES, DATASET

class Image:
    def __init__(self, name, path, path_to_db):
        self.PATH = os.path.join(path, name)
        self.name = self.filter_file(name)
        self.year = self.dismantle_name()[0]
        self.mission = self.dismantle_name()[1]
        self.id = self.dismantle_name()[2]
        self.angle = self.get_infos(path_to_db)
        self.date = [0,0]
        self.image = cv.imread(self.PATH)

    def filter_file(self, file_name):
        if file_name[-4:] == '.tif':
            name = file_name[:-4]
        elif file_name[-5:] == '.tiff':
            name = file_name[:-5]
        else:
            print("Le fichier n'est pas au format .tif/.tiff")
            return None
        return name
    
    def dismantle_name(self):
        i,j = 0, 0
        print(self.name)
        n = len(self.name)
        while i < n and self.name[i] != "_" :
            i += 1
        j = i+1
        while j < n and self.name[j] != "_" :
            j += 1
        year, mission, id = self.name[:i], self.name[i+1:j], self.name[j+1:]
        return int(year), mission, int(id)

    def get_infos(self, path_to_db) :
        db = pd.read_excel(path_to_db, str(self.year), header=0, skiprows=[1], usecols=['ID_mission', 'Numéro', 'angle'])
        db_mission = db[db["ID_mission"] == self.mission]
        row = db_mission[db_mission["Numéro"] == self.id]
        return -int(db_mission["angle"].values[0])

    def image_preparation(self, scale=0.1, inplace=False) :
        # Obtenir la taille de l'image
        (h, w) = self.image.shape[:2]

        # Définir le centre de rotation (au centre de l'image)
        center = (w // 2, h // 2)

        # Obtenir la matrice de transformation
        M = cv.getRotationMatrix2D(center, self.angle, 1.0)

        # Appliquer la rotation
        transformed = cv.warpAffine(self.image, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

        if scale:
            new_size = (round(w * scale), round(h * scale))
            transformed = cv.resize(transformed, new_size, interpolation=cv.INTER_AREA)
        
        if inplace:
            self.image = transformed
        return transformed
    
    def cleaning_borders(self, threshold=5):
        image = cv.imread(self.PATH, cv.IMREAD_GRAYSCALE)
    
        height, width = image.shape
        center_y, center_x = height // 2, width // 2
        k = 1

        # Fonction pour vérifier si le carré contient du noir
        def contains_black(x1, x2, y1, y2):
            return np.any(image[y1:y2, x1:x2] <= threshold)

        # Agrandir le carré jusqu'à atteindre un bord noir
        while True:
            x1, x2 = max(0, center_x - k), min(width, center_x + k)
            y1, y2 = max(0, center_y - k), min(height, center_y + k)

            if contains_black(x1, x2, y1, y2):  # Stop si un pixel noir est trouvé
                break
            k += 1
            if x1 == 0 or x2 == width or y1 == 0 or y2 == height:  # Stop si on atteint le bord
                break
        
        # On prend la dernière zone valide avant le noir
        x1, x2 = max(0, center_x - (k-1)), min(width, center_x + (k-1))
        y1, y2 = max(0, center_y - (k-1)), min(height, center_y + (k-1))

        cropped_image = image[y1:y2, x1:x2]
        return cropped_image

    


class Dataset:
    def __init__(self, directory, path_to_db='database.xlsx'):
        self.DIR = directory
        self.path_to_db = path_to_db
        self.images = self.load()
        self.names = [image.name for image in self.images]
    
    def load(self):
        NAMES,IMG = [],[]
        for filename in os.listdir(self.DIR):
            if '.tif' in filename: # obligatoire pour filtrer tout autre fichier/dossier dans ce répertoire
                image_path = os.path.join(self.DIR, filename)
                image = Image(filename, self.DIR, self.path_to_db)    
                IMG.append(image)
        return IMG
    
    def match(self, seuil_ransac=5.0):
        N = len(self.images)
        Hs = {}
        masks = {}
        for i in range(N):
            for j in range(i+1,N):
                img_i, img_j = self.images[i], self.images[j]
                kpt_i, kpt_j, matches = matching.init_matching_orb(img_i.image, img_j.image)
                matches = matching.filtre_distance(matches)
                H, mask = matching.ransac(kpt_i, kpt_j, matches, seuil_ransac)
                Hs[(self.names[i],self.names[j])] = H
                masks[(self.names[i],self.names[j])] = mask
        return Hs, masks
