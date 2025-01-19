import os
import cv2
import numpy as np
from ultralyticsplus import YOLO
from PIL import Image
from torchvision import transforms

from util import util

model = YOLO('kesimeg/yolov8n-clothing-detection')
target_label = 'clothing'

def crop_scale(image_path, output_path):
    # Charger l'image
    image = cv2.imread(image_path)

    # Détecter des objets dans l'image
    results = model(image)

    # Extraire les résultats de détection (boîtes englobantes, classes, etc.)
    if len(results[0].boxes) == 0:
        print(f"Object not detected for image")
        return

    detected = False
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        if label != target_label:
            continue
        detected = True

        x1, y1, x2, y2 = box.xyxy[0]  # Coordonnées de la boîte

        # Découper l'objet détecté
        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

        # Calculer le ratio de l'image
        h, w = cropped_image.shape[:2]
        target_size = 256
        scale = target_size / max(h, w)  # Trouver le facteur de mise à l'échelle

        # Redimensionner tout en conservant l'aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(cropped_image, (new_w, new_h))

        # Créer une image de fond blanche de 256x256
        padded_image = np.full((target_size, target_size, 3), 255, dtype=np.uint8)

        # Calculer les positions pour centrer l'image redimensionnée dans l'image de padding
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2

        # Copier l'image redimensionnée dans l'image de fond
        padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

        # Sauvegarder l'image
        cv2.imwrite(output_path, padded_image)
        print(f"Image sauvegardée sous {output_path}")

        break

    if not detected:
        print(f"Label {target_label} not detected for image")

def get_tensors(image_path):
    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")  # Convert to RGB if not already

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 (example size)
        transforms.ToTensor(),          # Convert to a tensor (0-1 range)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Apply transformations
    tensor_image = transform(image)  # Shape: [C, H, W]

    # Add batch dimension to create a 4D tensor
    return tensor_image.unsqueeze(0)  # Shape: [1, C, H, W]

def save_tensors(tensor_image_4d, name):
    im = util.tensor2im(tensor_image_4d)

    # Define the path to save the image
    results_folder = "./results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    save_path = os.path.join(results_folder, f"{name}.jpg")

    util.save_image(im, save_path, aspect_ratio=1.0)
