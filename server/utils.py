import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from models import create_model
from options.test_options import TestOptions
from util import util

def preprocess_image(model_yolo, image_bytes, output_path):
    # Convert bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Charger l'image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Détecter des objets dans l'image
    results = model_yolo(image)

    # Extraire les résultats de détection (boîtes englobantes, classes, etc.)
    if len(results[0].boxes) == 0:
        print(f"Objects not detected for image")
        return None

    detected = False
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        if label != 'clothing':
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

        # Create a PIL Image from the RGB image
        rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return pil_image

    if not detected:
        print(f"Clothes not detected for image")
        return None

def predict_professional_image(model_pic2pic, preprocessed_img, output_path):
    image_tensors = _get_tensors(preprocessed_img)

    # predict
    model_pic2pic.real = image_tensors.to(model_pic2pic.device)
    model_pic2pic.test()
    visuals = model_pic2pic.get_current_visuals() # results

    predicted_img = _get_image(visuals['fake'], output_path)
    return predicted_img

def _get_tensors(image):
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

def _get_image(tensors, output_path):
    im = util.tensor2im(tensors)

    # Define the path to save the image
    util.save_image(im, output_path, aspect_ratio=1.0)

    return Image.fromarray(im)

def load_model(opt):
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # init model
    model = create_model(opt)
    model.setup(opt)
    return model
