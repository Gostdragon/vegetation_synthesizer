import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skimage

from file_handler import save_gray_16

import config
import cv2
import glob


def blurr_generated_images(args, file_path, save_path, gaus=True):
    # blurr image with gaussian or median filter
    # works with both 8 and 16 bit
    paths = list(Path(file_path).glob('**/*_predict.png'))
    for path in paths:
        file_id = str(path).split('\\')[3].split('_')[0]

        input_image = cv2.imread(str(path))
        if gaus:
            blurred_image = cv2.GaussianBlur(input_image, (5, 5), 0)
        else:
            blurred_image = cv2.medianBlur(input_image, 5)
        cv2.imwrite(save_path + f'{file_id}_filtered.png', blurred_image)


def __clipping(v0, v1):
    if abs(v0 - v1) < 0.4:
        return v1
    return -1


def moving_average_filter(args):
    # moving average filter
    # consider the point in the kernel only if their difference is small enough
    paths = list(Path(config.PATH_TO_SKETCHES_OUTPUT + 'generated/').glob('**/*_predict.png'))
    denoms = [None, 1/1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9] # preprocessing

    for path in paths:
        file_id = str(path).split('\\')[2].split('_')[0]
        data = plt.imread(path)
        im = np.array(data, dtype=np.float32)
        width = im.shape[0]
        height = im.shape[1]

        res_im = np.zeros_like(im)
        for x in range(width):
            for y in range(height):
                nom = 0
                den = 0

                # 3x3 kernel
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if 0 <= x + i < width and 0 <= y + j < height:
                            val = __clipping(im[x][y], im[x + i][y + j])
                            nom, den = (nom + val, den + 1) if val >= 0 else (nom, den)

                res_im[x][y] = nom * denoms[den]

        res_im = np.uint16(res_im * 65535)

        if args.png16bits:
            save_gray_16(config.PATH_TO_SKETCHES_OUTPUT + f'generated/{file_id}_filtered.png', res_im)
        else:
            plt.imsave(config.PATH_TO_SKETCHES_OUTPUT + f'generated/{file_id}_filtered.png', res_im, cmap='terrain')


def adjust_bounding_box(x_min, x_max, y_min, y_max, x, y):
    # add point to AABB and calculate new one
    if x_max < x:
        x_max = x
    if x_min > x:
        x_min = x

    if y_max < y:
        y_max = y
    if y_min > y:
        y_min = y

    return x_min, x_max, y_min, y_max


def get_bounding_box_sketch(sketch):
    # calculate AABB via the rgb channel of the sketch
    # each pixel is either red, green or blue
    (width, height, _) = sketch.shape

    x_min = y_min = max(width, height)
    x_max = y_max = 0
    for x in range(width):
        for y in range(height):

            red = sketch[x][y][0] == 1
            green = sketch[x][y][1] == 1
            blue = sketch[x][y][2] == 1

            if red and not blue and not green:
                x_min, x_max, y_min, y_max = adjust_bounding_box(x_min, x_max, y_min, y_max, x, y)
            elif not red and blue and not green:
                x_min, x_max, y_min, y_max = adjust_bounding_box(x_min, x_max, y_min, y_max, x, y)
            elif not red and not blue and green:
                x_min, x_max, y_min, y_max = adjust_bounding_box(x_min, x_max, y_min, y_max, x, y)

    return x_min, x_max, y_min, y_max


def is_trash(veg_map, water_map):
    # returns if the maps contains invalid regions
    width = veg_map.shape[0]
    height = veg_map.shape[1]
    count_low_x_local_veg_pixel = np.count_nonzero(veg_map[0:20, ...] > 0)
    count_high_x_local_veg_pixel = np.count_nonzero(veg_map[width - 21:width - 1, ...] > 0)
    count_low_y_local_veg_pixel = np.count_nonzero(veg_map[:, 0:20, ...] > 0)
    count_high_y_local_veg_pixel = np.count_nonzero(veg_map[:, height - 21:height - 1, ...] > 0)

    count_low_x_local_water_pixel = np.count_nonzero(water_map[0:20, ...] > 0)
    count_high_x_local_water_pixel = np.count_nonzero(water_map[width - 21:width - 1, ...] > 0)
    count_low_y_local_water_pixel = np.count_nonzero(water_map[:, 0:20, ...] > 0)
    count_high_y_local_water_pixel = np.count_nonzero(water_map[:, height - 21:height - 1, ...] > 0)

    has_black_bar_veg_map = not (count_low_x_local_veg_pixel > 0 and count_high_x_local_veg_pixel > 0 and count_low_y_local_veg_pixel > 0 and count_high_y_local_veg_pixel > 0)
    has_black_bar_water_map = not (count_low_x_local_water_pixel > 0 and count_high_x_local_water_pixel > 0 and count_low_y_local_water_pixel > 0 and count_high_y_local_water_pixel > 0)

    if not has_black_bar_veg_map:
        return False
    if not has_black_bar_water_map:
        return False
    else:
        return True


def classify_realistic_minecraft_train_images():
    # categorises minecraft images if they be made of too much water or contains invalid regions
    file_pattern = str(config.PATH_TO_VEGETATION_OUTPUT + 'train_minecraft_realistic/*vegetation*.png')

    image_files = glob.glob(file_pattern)
    i = 0
    for file in image_files:
        image = plt.imread(file)
        w = image.shape[1]
        w = w // 2
        #height_map = image[:, :w, :][..., 1]
        veg_map = image[:, w:, :][..., 1]
        water_map = image[:, w:, :][..., 2]

        count_water_pixel = np.count_nonzero(water_map > 0)
        total_pixel = water_map.shape[0] * water_map.shape[1]
        if count_water_pixel > 0.5 * total_pixel:
            plt.imsave(config.PATH_TO_VEGETATION_OUTPUT + f'train_minecraft_realistic/water/{i}.png', image)
        elif is_trash(veg_map, water_map):
            plt.imsave(config.PATH_TO_VEGETATION_OUTPUT + f'train_minecraft_realistic/trash/{i}.png', image)
        else:
            plt.imsave(config.PATH_TO_VEGETATION_OUTPUT + f'train_minecraft_realistic/veg/{i}.png', image)

        i += 1


def convert_x_to_x_half_resolution():
    # rescales an image to half its resolution
    file_pattern = str('synthesizer/vegetation/own_heightmaps/heightmaps/*.png')
    image_files = glob.glob(file_pattern)
    i = 0
    for file in image_files:
        im = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
        if im.shape[0] != 256:
            im = cv2.pyrUp(np.array(im, dtype=np.float32), dstsize=(config.WIDTH, config.HEIGHT))
        w = im.shape[1]
        w = w // 2
        im_0 = im[:, :w]
        im_1 = im[:, w:]

        im_00 = im_0[:w, ...]
        im_01 = im_1[:w, ...]
        im_10 = im_0[w:, ...]
        im_11 = im_1[w:, ...]

        im_00 = np.uint16(im_00)
        im_01 = np.uint16(im_01)
        im_10 = np.uint16(im_10)
        im_11 = np.uint16(im_11)

        save_gray_16(f'synthesizer/vegetation/generated/vegetation_{i}_00.png', im_00)
        save_gray_16(f'synthesizer/vegetation/generated/vegetation_{i}_01.png', im_01)
        save_gray_16(f'synthesizer/vegetation/generated/vegetation_{i}_10.png', im_10)
        save_gray_16(f'synthesizer/vegetation/generated/vegetation_{i}_11.png', im_11)
        i += 1


def convert_x_half_to_x_resolution():
    # rescales an image to double its resolution
    file_pattern = str('synthesizer/vegetation/own_heightmaps/heightmaps/*.png')
    image_files = glob.glob(file_pattern)
    j = 3
    for i in range(0, len(image_files), 4):
        im_0 = cv2.imread(image_files[i], cv2.IMREAD_ANYDEPTH)
        im_1 = cv2.imread(image_files[i+1], cv2.IMREAD_ANYDEPTH)
        im_2 = cv2.imread(image_files[i+2], cv2.IMREAD_ANYDEPTH)
        im_3 = cv2.imread(image_files[i+3], cv2.IMREAD_ANYDEPTH)

        #result = np.zeros((im.shape[0] * 4, im.shape[1] * 4, 1), dtype=np.uint16)

        result_0 = np.concatenate([im_0, im_2], axis=0)
        result_1 = np.concatenate([im_1, im_3], axis=0)

        result = np.concatenate([result_0, result_1], axis=1)

        im = cv2.pyrDown(np.array(result, dtype=np.float32), dstsize=(config.WIDTH, config.HEIGHT))


        im = np.uint16(((im - np.amin(im)) / (np.amax(im) - np.amin(im))) * 65535)

        save_gray_16(f'synthesizer/vegetation/generated/vegetation_pyrUp_{j}.png', im)
        j += 1


def calculate_tertiary_map_vegetation_sketches(vegetation_map):
    # calculates tertiary vegetation density map out of vegetation sketch
    binary_vegetation_map = np.zeros((vegetation_map.shape[0], vegetation_map.shape[1], 1), dtype=np.float32)
    binary_vegetation_map[vegetation_map < 0.2] = 0
    binary_vegetation_map[0.7 <= vegetation_map] = 1
    mask = np.logical_and(0.2 <= vegetation_map, vegetation_map < 0.7)
    binary_vegetation_map[mask] = 0.5

    return binary_vegetation_map


def create_vertical_slices(args):
    # creates an vertical slice from images
    file_pattern = str('synthesizer/sketches/generated/*.png')
    image_files = glob.glob(file_pattern)
    for path in image_files:
        idx = str(path).split('\\')[1]
        f = 255 #start index
        till = 256 #end index
        if args.png16bits:
            data = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            data = (data - np.amin(data)) / (np.amax(data) - np.amin(data))
            z = np.array(data, dtype=np.float32)[..., f:till]
        else:
            data = plt.imread(path)
            z = np.array(data, dtype=np.float32)[..., f:till, 0:1]


        x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

        # show height map in 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(x, y, z)
        ax.scatter(x, y, z)
        #plt.plot(z)
        plt.title(str(path))

        plt.savefig(f'{config.PATH_TO_SKETCHES_OUTPUT}/generated/{idx}_asd.png', dpi=300)

        #plt.show()


def has_more_than_x_vegetation_percent_coverage(veg_map, percentage):
    # returns if the input image has more than x percent vegetation coverage
    percentage = percentage / 100.0
    veg_pixel = np.where(veg_map > 0)
    if len(veg_pixel[0]) > percentage * veg_map.shape[0] * veg_map.shape[1]:
        return True


def compute_erased_im(im, circles):
    # creates an image with circles like the circles in circles
    target = np.zeros(im.shape, dtype=np.float32)
    for (center_x, center_y, rad) in circles:
        target[skimage.draw.disk((center_x, center_y), radius=rad)] = 1

    im_c = im.copy()
    im_c[target == 1] = np.amin(im)

    return im_c


def create_circles(state, detailed_data):
    # creates 1 to 4 circles with random radius between config.MIN_CIRCLE_OFFSET and config.MAX_CIRCLE_OFFSET
    # no overlapping circles
    # calculated on detailed_data map
    circles = []
    for count in range(state.randint(1, 4)):
        # 1 to 3 circles per picture

        attempts = 0
        result = False
        while attempts < 10 and not result:
            rad_new = state.randint(config.MIN_CIRCLE_OFFSET, config.MAX_CIRCLE_OFFSET)

            cen_x_new = state.randint(1 + rad_new, detailed_data.shape[0] - 1 - rad_new)
            cen_y_new = state.randint(1 + rad_new, detailed_data.shape[1] - 1 - rad_new)

            distance_to_lower = min(cen_x_new, cen_y_new)
            distance_to_upper = min(detailed_data.shape[0] - cen_x_new, detailed_data.shape[1] - cen_y_new)

            dist_to_next_circle = detailed_data.shape[0]
            rad = 0
            for (cen_x, cen_y, r) in circles:
                dist = math.sqrt(math.pow(cen_x - cen_x_new, 2) + math.pow(cen_y - cen_y_new, 2))
                if dist < dist_to_next_circle:
                    dist_to_next_circle = dist
                    rad = r

            if rad_new + rad <= dist_to_next_circle and rad_new <= min(distance_to_lower, distance_to_upper):
                circles.append((cen_x_new, cen_y_new, rad_new))
                result = True

            attempts += 1

    return circles