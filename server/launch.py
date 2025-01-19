from models import create_model
from options.test_options import TestOptions
from server.utils import crop_scale, get_tensors, save_tensors

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

    # preprocess image
    image_path = "./datasets/t-shirts/1736765461.jpeg"
    scaled_path = "./datasets/t-shirts/1736765461_scaled.jpeg"
    crop_scale(image_path, scaled_path)
    image_tensors = get_tensors(scaled_path) # image to predict

    # predict
    model.real = image_tensors.to(model.device)
    model.test()
    visuals = model.get_current_visuals() # results

    # save results
    save_tensors(visuals['real'], 'real')
    save_tensors(visuals['fake'], 'fake')
