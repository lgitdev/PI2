import os
import requests
import pandas as pd
from PIL import Image
import io

def telechargerPhoto(photoID, mission, dossier, numero, orientation):
    """Télécharge une photo, ajuste son orientation et l'enregistre avec le numéro comme nom de fichier."""
    photoFile = f"{photoID}.tif"
    chemin = os.path.join(dossier, f"{numero}.tif")  # Nom basé uniquement sur "Numéro"

    # Construire l'URL
    url = f"https://data.geopf.fr/telechargement/download/pva/{mission}/{photoFile}"

    try:
        # Télécharger l'image
        reponse = requests.get(url, timeout=10)
        if reponse.status_code == 200:
            os.makedirs(dossier, exist_ok=True)

            # Charger l'image en mémoire avec PIL
            image = Image.open(io.BytesIO(reponse.content))

            # Appliquer la rotation
            image = image.rotate(-orientation, expand=True)  # Rotation pour correspondre à l'affichage du site

            # Sauvegarde avec le bon nom
            image.save(chemin, format="TIFF")
            print(f"Image téléchargée et orientée : {numero}.tif")
        else:
            print(f"Échec du téléchargement de {numero}.tif. Code HTTP : {reponse.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du téléchargement de {numero}.tif : {e}")

# Chemin du fichier Excel
fichierExcel = r"C:\Users\gindr\Documents\2024-2025\ESILV\Cours\S8\PI2\PI2\database.xlsx"
annee = '2007'
dossier = os.path.join(r"C:\Users\gindr\Documents\2024-2025\ESILV\Cours\S8\PI2\PI2", annee)

try:
    # Lire les données incluant l'orientation et le numéro
    data = pd.read_excel(fichierExcel, sheet_name=annee, 
                         usecols=["ID_mission", "ID_cliché", "Numéro", "Orientation du Nord (°)"], 
                         dtype={"ID_mission": str, "ID_cliché": str, "Numéro": str, "Orientation du Nord (°)": float}, 
                         engine='openpyxl')

    photos = data.to_numpy()

    # Téléchargement et rotation des images
    for cliche in photos:
        numero = cliche[2]  # Le fichier aura uniquement le "Numéro" comme nom
        orientation = float(cliche[3])  # Convertir en float si besoin
        telechargerPhoto(cliche[1], cliche[0], dossier, numero, orientation)

    print("Fin du script.")

except Exception as e:
    print(f"Erreur lors de la lecture du fichier Excel : {e}")
