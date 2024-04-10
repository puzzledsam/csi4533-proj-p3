import cv2
import numpy as np
import os
from PIL import Image
from utils import model, tools
import torch
from torchvision.ops import masks_to_boxes

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
    
    # Fichier d'images des personnes à identifier dans les images de caméra de surveillance
    person_imgs = ['targets/person_1.png', 'targets/person_2.png', 'targets/person_3.png', 'targets/person_4.png', 'targets/person_5.png']

    # Charger le modèle et appliquer les transformations à l'image
    seg_model, transforms = model.get_model()
    
    for person_img_path in person_imgs:
        
        print(f"Processing {person_img_path}...")
        
        # Charger l'image d'entrée (personne à identifier)
        person_img = cv2.imread(person_img_path)
        person_h, person_w, _ = person_img.shape

        # Réduire les couleurs de l'image
        div = 64 # Pixel intensity division factor
        reduced_person_img = person_img // div * div + div // 2
        
        person_histograms = getPersonHistograms(reduced_person_img, (0, 0, person_w, person_h))
        
        output_dir_full_path = os.path.join(output_path_dir, person_img_path.split("/")[-1])
        # Créer le dossier pour les résultats, si il n'existe pas 
        if not os.path.exists(output_dir_full_path):
            os.makedirs(output_dir_full_path)

        # Ouvrir les images et appliquer les transformations
        for root, dirs, _ in os.walk(source_path_dir):
            for subdir in dirs:
                full_dir = os.path.join(root, subdir)
                for file_name in os.listdir(full_dir):
                    
                    image_path = os.path.join(full_dir, file_name)
                    image = Image.open(image_path)
                    transformed_img = transforms(image)
                    
                    # Effectuer l'inférence sur l'image transformée sans calculer les gradients
                    with torch.no_grad():
                        output = seg_model([transformed_img])

                    # Traiter le résultat de l'inférence
                    result = tools.process_inference(output,image)
                    
                    result_mask = np.load('output/saved_masks.npy') # Charger le masques détectés lors de l'inférence
                    result_boxes = masks_to_boxes(torch.from_numpy(result_mask)) # Transformer les masques en bounding box
                    
                    # Charger l'image que l'on veux comparer avec l'image de notre personne
                    candidate_img = cv2.imread(image_path)
                    
                    # Réduire les couleurs de l'image
                    reduced_candidate_img = candidate_img // div * div + div // 2
                    
                    best_match = None
                    best_match_score = -1
                    for box in result_boxes:
                        # Convertir boite englobante pytorch en tuple d'entiers
                        box_list = [int(val) for val in box.tolist()]

                        candidate_histograms = getPersonHistograms(reduced_candidate_img, (box_list[0], box_list[1], box_list[2]-box_list[0], box_list[3]-box_list[1]))
                        
                        # Faire la comparaison du candidat et de la personne recherchée
                        comparison_result = comparePersonHistograms(person_histograms, candidate_histograms)
                        if comparison_result is not None:
                            # Si le score est plus que (ou égal à) 2.2, c'est probablement la personne qu'on cherche,
                            # mais on donne la chance de voir si il y a un meilleur candidat dans l'image
                            if comparison_result >= 2.2 and comparison_result > best_match_score:
                                #print(f"Ding! Score of {comparison_result} at {box_list}")
                                best_match_score = comparison_result
                                best_match = box_list
                                
                    if best_match is not None:
                        print(f"Possible match for {person_img_path} in {subdir}/{file_name}, result of comparison has score of {best_match_score}: {best_match}")
                        annotated_candidate_img = cv2.rectangle(candidate_img, (best_match[0], best_match[1]),  (best_match[2], best_match[3]), (255, 0, 0), 2)
                        cv2.imwrite(os.path.join(output_dir_full_path, file_name), annotated_candidate_img)
        print(f"\n{person_img_path} finished.\n")
    print("All Done!")