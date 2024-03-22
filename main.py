import os
from PIL import Image
from utils import model, tools
import torch

# Point d'entrée principal du script
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