import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import glob
from pathlib import Path

import config
from file_handler import save_gray_16
from tf_pix2pix.model_original_pix2pix import fit, generate_images_model


def load(args, image_file):
    image = tf.io.read_file(image_file)
    if args.png16bits:
        image = tf.io.decode_png(image, dtype=tf.uint16)
    else:
        image = tf.io.decode_png(image)

    w = tf.shape(image)[1]
    if w == 512:
        w = w // 2
        input_height_image = image[:, :w, :]
        real_image = image[:, w:, :]
        input_vegetation_image = input_height_image # dummy image
        has_veg = False

    else:
        w = w // 3
        input_height_image = image[:, :w, :]
        input_vegetation_image = image[:, w:w*2, :]
        real_image = image[:, w*2:, :]
        input_vegetation_image = input_vegetation_image
        has_veg = True

    input_height_image = tf.image.convert_image_dtype(input_height_image, tf.float32)
    input_vegetation_image = tf.image.convert_image_dtype(input_vegetation_image, tf.float32)
    real_image = tf.image.convert_image_dtype(real_image, tf.float32)

    return input_height_image, input_vegetation_image, real_image, has_veg


def resize(args, input_image, vegetation_image, real_image, height, width, has_veg):
    if has_veg:
        vegetation_image = tf.image.resize(vegetation_image, [height, width], method=tf.image.ResizeMethod.BICUBIC)
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.BICUBIC)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.BICUBIC)

    return input_image, vegetation_image, real_image


def random_crop(input_image, vegetation_image, real_image, has_veg):
    if has_veg:
        stacked_image = tf.stack([input_image, vegetation_image, real_image], axis=0)
    else:
        stacked_image = tf.stack([input_image, real_image], axis=0)

    if tf.shape(stacked_image)[3] == 1:
        if has_veg:
            cropped_image = tf.image.random_crop(stacked_image, size=[3, config.HEIGHT, config.WIDTH, 1])
        else:
            cropped_image = tf.image.random_crop(stacked_image, size=[2, config.HEIGHT, config.WIDTH, 1])
    else:
        if has_veg:
            cropped_image = tf.image.random_crop(stacked_image, size=[3, config.HEIGHT, config.WIDTH, 4])
        else:
            cropped_image = tf.image.random_crop(stacked_image, size=[2, config.HEIGHT, config.WIDTH, 4])

    if has_veg:
        return cropped_image[0], cropped_image[1], cropped_image[2]
    else:
        return cropped_image[0], cropped_image[1], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, vegetation_image, real_image, has_veg):
    if has_veg:
        vegetation_image = (vegetation_image * 2) - 1
    input_image = (input_image * 2) - 1
    real_image = (real_image * 2) - 1

    return input_image, vegetation_image, real_image


def random_jitter(args, input_image, vegetation_image, real_image, has_veg):
    # Resizing to 286x286
    input_image, vegetation_image, real_image = resize(args, input_image, vegetation_image, real_image,
                                                       config.HEIGHT_PREPROC, config.HEIGHT_PREPROC, has_veg)

    if has_veg:
        vegetation_image = (vegetation_image / tf.reduce_max(vegetation_image))

    input_image = (input_image / tf.reduce_max(input_image))
    real_image = (real_image / tf.reduce_max(real_image))

    # Random cropping back to 256x256
    input_image, vegetation_image, real_image = random_crop(input_image, vegetation_image, real_image, has_veg)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        if has_veg:
            vegetation_image = tf.image.flip_left_right(vegetation_image)
        input_image = tf.image.flip_left_right(input_image)

        real_image = tf.image.flip_left_right(real_image)

    return input_image, vegetation_image, real_image


def load_image_train(args, image_file):
    input_image, vegetation_image, real_image, has_veg = load(args, image_file)
    input_image, vegetation_image, real_image = random_jitter(args, input_image, vegetation_image, real_image, has_veg)
    input_image, vegetation_image, real_image = normalize(input_image, vegetation_image, real_image, has_veg)

    if has_veg:
        return (tf.stack((input_image[..., 0], vegetation_image[..., 0],), axis=2),
                tf.expand_dims((real_image[..., 0]), axis=2))
    else:
        return (tf.expand_dims((input_image[..., 0]), axis=2)), tf.expand_dims((real_image[..., 0]), axis=2)


def load_image_test(args, image_file):
    input_image, vegetation_image, real_image, has_veg = load(args, image_file)

    if has_veg:
        vegetation_image = (vegetation_image / tf.reduce_max(vegetation_image))

    input_image = (input_image / tf.reduce_max(input_image))
    real_image = (real_image / tf.reduce_max(real_image))

    input_image, vegetation_image, real_image = normalize(input_image, vegetation_image, real_image, has_veg)

    if has_veg:
        return (tf.stack((input_image[..., 0], vegetation_image[..., 0],), axis=2),
                tf.expand_dims((real_image[..., 0]), axis=2))
    else:
        return (tf.expand_dims((input_image[..., 0]), axis=2)), tf.expand_dims((real_image[..., 0]), axis=2)


def load_data(args):
    print("Data loading")

    file_pattern = str(config.PATH_TO_VEGETATION_OUTPUT + 'train_' + config.COUNTRY + '/*.png')

    # Load and preprocess the data using NumPy arrays
    image_files = glob.glob(file_pattern)
    dataset = tf.data.Dataset.from_tensor_slices(image_files)

    train_dataset = (dataset.skip(100)
                     .shuffle(config.BUFFERSIZE)
                     .map(lambda x: load_image_train(args, x), num_parallel_calls=tf.data.AUTOTUNE)
                     .repeat()
                     .batch(config.BATCH_SIZE)
                     .prefetch(tf.data.AUTOTUNE)
                     )
    test_dataset = (dataset.take(100)
                    # .shuffle(config.BUFFERSIZE)
                    .map(lambda x: load_image_test(args, x), num_parallel_calls=tf.data.AUTOTUNE)
                    .repeat()
                    .batch(1)  # config.BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE)
                    )

    print("Data preprocessed")

    return train_dataset, test_dataset


def load_own_images(args, image_file):
    input_image, vegetation_image, real_image, has_veg = load(args, image_file)
    if has_veg:
        vegetation_image = (vegetation_image / tf.reduce_max(vegetation_image))

    input_image = (input_image / tf.reduce_max(input_image))
    real_image = (real_image / tf.reduce_max(real_image))

    input_image, vegetation_image, real_image = normalize(input_image, vegetation_image, real_image, has_veg)

    if has_veg:
        return (tf.stack((input_image[..., 0], vegetation_image[..., 0],), axis=2),
                tf.expand_dims((real_image[..., 0]), axis=2))
    else:
        return (tf.expand_dims((input_image[..., 0]), axis=2)), tf.expand_dims((real_image[..., 0]), axis=2)




def load_own_sketches(args, veg_path):
    paths = list(Path(veg_path).glob('**/*.png'))

    i = 0
    if 1:
        # input for generator creation
        for i in range(0, len(paths), 2):
            height_path = paths[i]
            veg_path = paths[i + 1]

            height_map = plt.imread(height_path)
            veg_map = plt.imread(veg_path)
            height_map = np.array(height_map, dtype=np.float32)
            veg_map = np.array(veg_map, dtype=np.float32)
            target_map = np.zeros((height_map.shape[0], height_map.shape[1]))  # dummy target

            input_image = np.concatenate([height_map, veg_map, target_map], axis=1)

            if args.png16bits:
                input_image = np.squeeze(np.uint16(input_image * 65535))
                save_gray_16(config.PATH_TO_VEGETATION_OUTPUT + f'own_heightmaps/input_generator/{i}.png', input_image)
            else:
                input_image = np.squeeze(np.uint8(input_image * 127.5 + 127.5))
                plt.imsave(config.PATH_TO_VEGETATION_OUTPUT + f'own_heightmaps/input_generator/{i}.png', input_image,
                           cmap='gray')

            i += 1


    # Load dataset with tf
    image_files = glob.glob(config.PATH_TO_VEGETATION_OUTPUT + f'own_heightmaps/input_generator/*.png')
    dataset = tf.data.Dataset.from_tensor_slices(image_files)

    dataset = (dataset
               .map(lambda x: load_own_images(args, x), num_parallel_calls=tf.data.AUTOTUNE).repeat()
               .batch(1)
               .prefetch(tf.data.AUTOTUNE)
               )

    return dataset, len(image_files)



def generate_images(args, generator, source, target, idx, path_pic):
    if args.png16bits:
        predicted = generator(source[0:1, ...], training=False)

        if source.shape[3] == 1:
            im_source = np.squeeze(np.uint16(source[0, ...] * 32767.5 + 32767.5), axis=-1)
        else:
            source_height = np.uint16(source[0, :, :, 0] * 32767.5 + 32767.5)
            source_vegetation = np.uint16(source[0, :, :, 1] * 32767.5 + 32767.5)
            im_source = np.concatenate([source_height, source_vegetation], axis=-1)

        predicted = np.squeeze(np.uint16(predicted[0, ...] * 32767.5 + 32767.5), axis=-1)
        target = np.squeeze(np.uint16(target[0, ...] * 32767.5 + 32767.5), axis=-1)

        save_gray_16(path_pic + f'{idx}_input.png', im_source)
        save_gray_16(path_pic + f'{idx}_predict.png', predicted)
        save_gray_16(path_pic + f'{idx}_target.png', target)

    else:
        predicted = generator(source[0:1, ...], training=False)

        target = np.squeeze(np.uint8(target[0, ...] * 127.5 + 127.5), axis=-1)
        im_p = np.squeeze(np.uint8(predicted[0, ...] * 127.5 + 127.5), axis=-1)
        im_source = np.squeeze(np.uint8(source[0:1, ...][0, ...] * 127.5 + 127.5), axis=-1)

        plt.imsave(path_pic + f'{idx}_predict.png', im_p, cmap='gray')
        plt.imsave(path_pic + f'{idx}_target.png', target, cmap='gray')
        plt.imsave(path_pic + f'{idx}_input.png', im_source, cmap='gray')


def train_gan(args):
    train_dataset, test_dataset = load_data(args)
    fit(args, generate_images, 2, train_dataset, test_dataset, config.PATH_TO_VEGETATION_MILESTONE,
        config.PATH_TO_VEGETATION_MODEL_MILESTONE, steps=25000)


def create_images(args, c):
    _, test_dataset = load_data(args)
    generate_images_model(args, generate_images, 2, test_dataset, config.PATH_TO_VEGETATION_OUTPUT + '/generated/',
                          config.PATH_TO_VEGETATION_MODEL_MILESTONE, c)


def create_vegetation_from_heightmap(args, veg_path):
    test_dataset, c = load_own_sketches(args, veg_path)
    generate_images_model(args, generate_images, 2, test_dataset, config.PATH_TO_VEGETATION_OUTPUT
                          + 'generated/', config.PATH_TO_VEGETATION_MODEL_MILESTONE, c)
