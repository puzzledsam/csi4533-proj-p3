import cv2
import numpy as np
import os
from PIL import Image
from utils import model, tools
import torch

### FONCTIONS ###

def getPersonHistograms(source_img, person_rect):
    '''
        Retourne l'histogramme complet et le demi-histogramme
        pour la personne délimitée par le rectangle dans l'image source.
    '''
    
    # Obtenir les masques pour une personne et sa moitié supérieure
    masks = []
    for i in [1,2]:
        x, y, w, h = person_rect
        
        if h < 40 or w < 25:
            # On veut ignorer les petits régions
            break
        
        h = int(h / i) # Première iteration prends le masque complet, le deuxième ne prends que la moitiée supérieure
        
        cv2.rectangle(source_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        mask = np.zeros(source_img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255), -1)
        masks.append(mask)
        
    # Calculer histogrammes pour chaque masque
    histograms = []
    for mask in masks:
        hist = cv2.calcHist([source_img], [0, 1, 2], mask, [25] * 3, [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
        
    
    return histograms

def comparePersonHistograms(person1, person2):
    '''
        Comparer jusqu'à 4 histogrammes ensemble (jusqu'à 2 par personne) pour
        trouver un score de correspondance et renvoyer le score maximum.
    '''
    
    scores = []
    for i in range(len(person1)):
        for j in range(len(person2)):
            # Compare chaque combination des histogrammes avec la méthode d'intersection
            scores.append(cv2.compareHist(person1[i], person2[j], cv2.HISTCMP_INTERSECT))
    
    return max(scores, default=None)

### Point d'entrée principal du script ###

if __name__ == "__main__":
    
    torch_device_name = "cpu"
    if torch.backends.mps.is_available():
        # Pour les Macs avec Apple Silicon
        torch_device_name = "mps"
    elif torch.cuda.is_available():
        # Pour les GPU Nvidia
        torch_device_name = "cuda"
        
    torch_device = torch.device(torch_device_name)
    print(f"Using {torch_device} torch device for inference")

    # Définir les répertoires source et de sortie
    source_path_dir = "images"
    output_path_dir = "output"
    #image_name = "sample_2.png"

    # Charger le modèle et appliquer les transformations à l'image
    seg_model, transforms = model.get_model()

    # Ouvrir les images et appliquer les transformations
    for root, dirs, _ in os.walk(source_path_dir):
        for subdir in dirs:
            full_dir = os.path.join(root, subdir)
            for file_name in os.listdir(full_dir):
                
                output_dir_full_path = os.path.join(output_path_dir, subdir)
                # Créer le dossier pour les résultats, si il n'existe pas 
                if not os.path.exists(output_dir_full_path):
                    os.makedirs(output_dir_full_path)
                
                image_path = os.path.join(full_dir, file_name)
                image = Image.open(image_path)
                transformed_img = transforms(image)
                

                # Effectuer l'inférence sur l'image transformée sans calculer les gradients
                with torch.no_grad():
                    output = seg_model([transformed_img])

                # Traiter le résultat de l'inférence
                result = tools.process_inference(output,image)
                result.save(os.path.join(output_dir_full_path, file_name))
                # result.show()