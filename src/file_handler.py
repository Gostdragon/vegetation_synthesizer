import os

import numpy as np
import matplotlib.pyplot as plt
import png

import config


def save_sketches_picture_from_training_data():
    print_training_pictures(config.PATH_TO_TRAINING_DATA + config.COUNTRY + '/sketches/unfiltered.npz',
                            'sketch', config.PATH_TO_SKETCHES_OUTPUT, is_sketch=True)


def save_levelset_pictures_from_training_data():
    print_training_pictures(config.PATH_TO_TRAINING_DATA + config.COUNTRY + '/levelset/levelset.npz',
                            'levelset', config.PATH_TO_LEVELSET_OUTPUT)


def save_eraser_pictures_from_training_data():
    print_training_pictures(config.PATH_TO_TRAINING_DATA + config.COUNTRY + '/eraser/unfiltered.npz',
                            'eraser', config.PATH_TO_ERASER_OUTPUT)


def save_vegetation_pictures_from_training_data():
    print_training_pictures(config.PATH_TO_TRAINING_DATA + config.COUNTRY + '/vegetation/unfiltered.npz',
                            'vegetation', config.PATH_TO_VEGETATION_OUTPUT, is_veg=True)


def dir_exists_and_is_empty(path):
    if not os.path.exists(str(path)):
        os.mkdir(str(path))
    else:
        if len(os.listdir(str(path))) != 0:
            print("dir " + str(path) + "is not empty!")
            return False
    return True


def print_training_pictures(p_dataset, name_input, p_output, is_sketch=False, is_veg=False):
    data = np.load(p_dataset)
    input_height = data['x']
    output = data['y']

    dir = p_output + 'train_' + config.COUNTRY + '/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    if is_sketch:
        for i in range(input_height.shape[0]):
            if i > 100:
                return
            x = input_height[i]
            y = output[i]

            if is_sketch:
                height_3_dim = np.ones((config.WIDTH, config.HEIGHT, 3), dtype=np.float32)
                height_3_dim[..., 0] = (y[..., 0] + 1) / 2
                height_3_dim[..., 1] = (y[..., 0] + 1) / 2
                height_3_dim[..., 2] = (y[..., 0] + 1) / 2
                y = height_3_dim

            im = np.concatenate([x, y], axis=1).squeeze()
            plt.imsave(p_output + f'train_{config.COUNTRY}/{name_input}_{i}.png', im)

    if is_sketch:
        # sketch
        for i in range(output.shape[0]):
            x = input_height[i]
            y = output[i]

            height_3_dim = np.ones((config.WIDTH, config.HEIGHT, 3), dtype=np.float32)
            height_3_dim[..., 0] = (y[..., 0] + 1) / 2
            height_3_dim[..., 1] = (y[..., 0] + 1) / 2
            height_3_dim[..., 2] = (y[..., 0] + 1) / 2
            y = height_3_dim

            im = np.concatenate([x, y], axis=1).squeeze()
            im = np.uint16(im * 65535)
            save_rgb_16(p_output + f'train_{config.COUNTRY}/{name_input}_{i}.png', im)

    elif is_veg:
        # vegetation
        # changing depending on used dataset
        #input_vegetation = data['z']
        erode = data['erode']

        for i in range(input_height.shape[0]):
            height_map = input_height[i]
            output_veg_map = output[i]
            #input_vegetation_map = input_vegetation[i]
            #erode_map = erode[i]

            im = np.concatenate([height_map, output_veg_map], axis=1).squeeze()
            im = np.uint16(im * 65535)
            save_gray_16(p_output + 'train_' + config.COUNTRY + f'/{name_input}_{i}.png', im)

    else:
        # levelset or eraser
        for i in range(output.shape[0]):
            x = input_height[i]
            y = output[i]
            im = np.concatenate([x, y], axis=1).squeeze()

            im = np.uint16(im * 65535)
            save_gray_16(p_output + 'train_' + config.COUNTRY + f'/{name_input}_{i}.png', im)


def save_gray_16(path, im):
    with open(path, 'wb') as f:
        writer = png.Writer(width=im.shape[1], height=im.shape[0], bitdepth=16,
                            greyscale=True)
        # Convert z to the Python list of lists expected by
        # the png writer.
        z2list = im.tolist()
        writer.write(f, z2list)


def save_rgb_16(path, im):
    with open(path, 'wb') as f:
        writer = png.Writer(width=im.shape[1], height=im.shape[0], bitdepth=16,
                            greyscale=False)
        # Convert z to the Python list of lists expected by
        # the png writer.
        z2list = im.reshape(-1, im.shape[1] * im.shape[2]).tolist()
        writer.write(f, z2list)
