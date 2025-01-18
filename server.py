import os
from models import create_model
from options.test_options import TestOptions
from util import util
from torchvision import transforms
from PIL import Image

def get_tensors():
    # Path to your image
    image_path = "./datasets/t-shirts/1736765461.jpeg"

    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")  # Convert to RGB if not already

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 (example size)
        transforms.ToTensor(),          # Convert to a tensor (0-1 range)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Apply transformations
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

if __name__ == '__main__':
    opt = TestOptions().parse()

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # init model
    model = create_model(opt)
    model.setup(opt)

    image_tensors = get_tensors() # image to predict
    # run model
    model.real = image_tensors.to(model.device)
    model.test()
    visuals = model.get_current_visuals() # results

    # save results
    save_tensors(visuals['real'], 'real')
    save_tensors(visuals['fake'], 'fake')
