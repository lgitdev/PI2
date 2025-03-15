import cv2 as cv
import pandas as pd


class Images:
    def __init__(self, nom):
        self.name = nom
        self.img = self.get_image()
        self.scale = self.get_scale()

    def get_image(self):
        return cv.imread(self.name)
    
    def get_year_mission_id(self):
        i = 0
        while i<len(self.name) and self.name[i] != '_' :
            i+=1
        if i == len(self.name) :
            return f"Le nom {self.name} n'est pas valide."
        year = self.name[:i]
        i+=1
        j=i
        while j<len(self.name) and self.name[j] != '_' :
            j+=1
        if j == len(self.name) :
            return f"Le nom {self.name} n'est pas valide."
        mission = self.name[i:j]
        j+=1
        k=j
        while k<len(self.name) and self.name[k] != '.' :
            k+=1
        if k == len(self.name) :
            return f"Le nom {self.name} n'est pas valide."
        id = self.name[j:k]
        return year, mission, id

    def get_scale(self):
        fichierExcel = 'database.xlsx' # Chemin vers l'excel
        year, mission, id = self.get_year_mission_id()
        data = pd.read_excel(fichierExcel, year, usecols=[0,2,4], dtype={'ID_mission':str, 'Numéro':str, 'echelle':float}, skiprows=[1])
        
        echelle_value = data.loc[(data['ID_mission'] == mission) & (data['Numéro'] == id), 'echelle'].values
        if len(echelle_value) == 0 :
            return f"Aucune ligne ne correspond à {self.name} dans database.xlsx"
        return echelle_value

    def get_angle(self):
        fichierExcel = 'database.xlsx' # Chemin vers l'excel
        year, mission, id = self.get_year_mission_id()
        data = pd.read_excel(fichierExcel, year, usecols=[0,2,6], dtype={'ID_mission':str, 'Numéro':str, 'angle':int}, skiprows=[1])
        
        angle_value = data.loc[(data['ID_mission'] == mission) & (data['Numéro'] == id), 'angle'].values
        if len(angle_value) == 0 :
            return f"Aucune ligne ne correspond à {self.name} dans database.xlsx"
        return angle_value
    
