import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pathlib as Path
import glob
from tf_pix2pix.model_original_pix2pix import fit, generate_images_model

from file_handler import save_gray_16
import config


def load(args, image_file):
    image = tf.io.read_file(image_file)

    if args.png16bits:
        image = tf.io.decode_png(image, dtype=tf.uint16)
    else:
        image = tf.io.decode_png(image)

    w = tf.shape(image)[1]
    w = w // 2
    real_image = image[:, w:, :]
    input_image  = image[:, :w, :]

    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    real_image = tf.image.convert_image_dtype(real_image, tf.float32)

    return input_image, real_image


def resize(args, input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, config.HEIGHT, config.WIDTH, 1])

    return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image * 2) - 1
    real_image = (real_image * 2) - 1

    return input_image, real_image


def random_jitter(args, input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(args, input_image, real_image, config.HEIGHT_PREPROC, config.HEIGHT_PREPROC)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(args, image_file):
    input_image, real_image = load(args, image_file)
    input_image, real_image = random_jitter(args, input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return tf.expand_dims(input_image[..., 0], axis=2), tf.expand_dims(real_image[..., 0], axis=2)


def load_image_test(args, image_file):
    input_image, real_image = load(args, image_file)
    # input_image, real_image = resize(args, input_image, real_image, config.HEIGHT, config.WIDTH)
    input_image = (input_image / tf.reduce_max(input_image))
    real_image = (real_image / tf.reduce_max(real_image))
    input_image, real_image = normalize(input_image, real_image)

    return tf.expand_dims(input_image[..., 0], axis=2), tf.expand_dims(real_image[..., 0], axis=2)


def load_data(args):
    file_pattern = str(config.PATH_TO_ERASER_OUTPUT + 'train_' + config.COUNTRY + '/*.png')

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


def load_own_erased(args, veg_path):
    paths = list(Path(veg_path).glob('**/*.png'))

    i = 0
    if 1:
        for i in range(0, len(paths), 1):
            erased_path = paths[i]

            erased_map = plt.imread(erased_path)
            erased_map = np.array(erased_map, dtype=np.float32)
            target_map = np.zeros((erased_map.shape[0], erased_map.shape[1]))

            input_image = np.concatenate([erased_map, target_map], axis=1)

            if args.png16bits:
                input_image = np.squeeze(np.uint16(input_image * 65535))
                save_gray_16(config.PATH_TO_VEGETATION_OUTPUT + f'own_heightmaps/input_generator/{i}.png', input_image)
            else:
                #input_image_rgb = np.squeeze(np.uint8(input_image_rgb * 127.5 + 127.5))
                plt.imsave(config.PATH_TO_VEGETATION_OUTPUT + f'own_heightmaps/input_generator/{i}.png', input_image, cmap='terrain')
            i += 1

    # Load dataset with tf
    image_files = glob.glob(config.PATH_TO_VEGETATION_OUTPUT + f'own_heightmaps/input_generator/*.png')
    dataset = tf.data.Dataset.from_tensor_slices(image_files)

    dataset = (dataset
               .map(lambda x: load_image_test(args, x), num_parallel_calls=tf.data.AUTOTUNE).repeat()
               .batch(1)
               .prefetch(tf.data.AUTOTUNE)
               )
    return dataset, len(image_files)


def generate_images(args, generator, source, target, idx, path_pic):
    if args.png16bits:
        predicted = generator(source[0:1, ...], training=False)

        im_p = np.squeeze(np.uint16(predicted[0, ...] * 32767.5 + 32767.5), axis=-1)
        target = np.squeeze(np.uint16(target[0, ...] * 32767.5 + 32767.5), axis=-1)
        im_source = np.squeeze(np.uint16(source[0:1, ...][0, ...] * 32767.5 + 32767.5), axis=-1)

        save_gray_16(path_pic + f'{idx}_predict.png', im_p)
        save_gray_16(path_pic + f'{idx}_target.png', target)
        save_gray_16(path_pic + f'{idx}_source.png', im_source)

    else:
        predicted = generator(source[0:1, ...], training=False)

        target = np.uint8(target[0, ...] * 127.5 + 127.5)
        im_p = np.uint8(predicted[0, ...] * 127.5 + 127.5)
        im_source = np.uint8(source[0:1, ...][0, ...] * 127.5 + 127.5)

        plt.imsave(path_pic + f'{idx}_predict.png', np.squeeze(im_p, axis=-1), cmap='terrain')
        plt.imsave(path_pic + f'{idx}_target.png', np.squeeze(target, axis=-1), cmap='gray')
        plt.imsave(path_pic + f'{idx}_source.png', np.squeeze(im_source, axis=-1))


def train_gan(args):
    train_dataset, test_dataset = load_data(args)
    fit(args, generate_images, 1, train_dataset, test_dataset, config.PATH_TO_ERASER_MILESTONE,
        config.PATH_TO_ERASER_MODEL_MILESTONE, steps=60000)


def create_images(args, c):
    _, test_dataset = load_data(args)
    generate_images_model(args, generate_images, 1, test_dataset, config.PATH_TO_ERASER_OUTPUT
                          + 'generated/', config.PATH_TO_ERASER_MODEL_MILESTONE, c)

def create_eraser_from_erased_image(args, veg_path, in_channels):
    test_dataset, c = load_own_erased(args, veg_path)
    generate_images_model(args, generate_images, in_channels, test_dataset, config.PATH_TO_VEGETATION_OUTPUT
                          + 'generated/', config.PATH_TO_VEGETATION_MODEL_MILESTONE, c)
