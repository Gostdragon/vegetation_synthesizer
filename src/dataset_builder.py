import os

import time
from pysheds.grid import Grid
import numpy as np
from skimage.util import view_as_blocks
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import georasters as gr
import cv2
from skimage.morphology import skeletonize
import concurrent.futures
import glob

import config
from utils import compute_erased_im, create_circles


def extract_patches_from_raster():
    count = 0
    start = time.time()
    for raster_file in Path(config.PATH_TO_INPUT).glob('**/*.tif'):
        data = gr.from_file(str(raster_file))
        data = data.resize((int(data.shape[0] / config.WIDTH) * config.WIDTH,
            (int(data.shape[1] / config.HEIGHT) * config.HEIGHT)))
        raster_blocks = view_as_blocks(data.raster, (config.WIDTH, config.HEIGHT))

        for i in range(raster_blocks.shape[0]):
            for j in range(raster_blocks.shape[1]):
                raster_data = raster_blocks[i, j]

                src = cv2.pyrDown(
                    raster_data,
                    dstsize=(
                        raster_data.shape[1] // 2,
                        raster_data.shape[0] // 2))

                data_out_downsampled = gr.GeoRaster(
                    src,
                    data.geot,
                    nodata_value=data.nodata_value,
                    projection=data.projection,
                    datatype=data.datatype,
                    fill_value=0
                )

                data_out_downsampled.to_tiff(
                    config.PATH_TO_DOWNSAMPLED_DATA + 'data_q' + str(count) + str(i) + str(j))

                data_out = gr.GeoRaster(
                    raster_data,
                    data.geot,
                    nodata_value=data.nodata_value,
                    projection=data.projection,
                    datatype=data.datatype,
                    fill_value=0
                )
                data_out.to_tiff(
                    config.PATH_TO_DATA + 'data_q' + str(count) + str(i) + str(j))
                count += 1

                if count % 1000 == 0:
                    end = time.time()
                    print(f'time count: {end - start}')


def compute_rivers(tiff_image):
    grid = Grid.from_raster(str(tiff_image), data_name='dem')
    dem = grid.read_raster(str(tiff_image))
    depressions = grid.detect_depressions(dem)

    flooded_dem = grid.fill_depressions(dem)
    flats = grid.detect_flats(flooded_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Compute flow direction based on corrected DEM
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    dir = grid.flowdir(inflated_dem, dirmap=dirmap)
    # Compute flow accumulation based on computed flow direction
    acc = grid.accumulation(dir, dirmap=dirmap)
    downsampled_rivers = np.log(grid.view(acc) + 1)
    upsampled_depressions = cv2.pyrUp(
        np.array(depressions, dtype=np.uint8),
        dstsize=(config.WIDTH, config.HEIGHT))

    upsampled_rivers = cv2.pyrUp(
        downsampled_rivers,
        dstsize=(config.WIDTH, config.HEIGHT))
    upsampled_rivers = (upsampled_rivers - np.amin(upsampled_rivers)) / \
                       (np.amax(upsampled_rivers) - np.amin(upsampled_rivers))
    upsampled_rivers = np.array(upsampled_rivers * 255, dtype=np.uint8)
    _, thresholded_river = cv2.threshold(upsampled_rivers, 127, 255, cv2.THRESH_BINARY)
    thresholded_river[thresholded_river == 255] = 1
    skeletonized_rivers = skeletonize(thresholded_river)

    return np.expand_dims(skeletonized_rivers, axis=-1), np.expand_dims(upsampled_depressions, axis=-1)


def compute_ridges(tiff_image):
    grid = Grid.from_raster(str(tiff_image), data_name='dem')
    dem = grid.read_raster(str(tiff_image))
    dem = dem.max() - dem

    peaks = grid.detect_depressions(dem)
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    flats = grid.detect_flats(flooded_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Compute flow direction based on corrected DEM
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    dir = grid.flowdir(inflated_dem, dirmap=dirmap)
    # Compute flow accumulation based on computed flow direction
    acc = grid.accumulation(dir, dirmap=dirmap)
    downsampled_ridges = np.log(grid.view(acc) + 1)
    upsampled_peaks = cv2.pyrUp(
        np.array(peaks, dtype=np.uint8),
        dstsize=(config.WIDTH, config.HEIGHT))
    upsampled_ridges = cv2.pyrUp(
        downsampled_ridges,
        dstsize=(config.WIDTH, config.HEIGHT))
    upsampled_ridges = (upsampled_ridges - np.amin(upsampled_ridges)) / \
                       (np.amax(upsampled_ridges) - np.amin(upsampled_ridges))
    upsampled_ridges = np.array(upsampled_ridges * 255, dtype=np.uint8)
    _, thresholded_ridges = cv2.threshold(upsampled_ridges, 150, 255, cv2.THRESH_BINARY)
    thresholded_ridges[thresholded_ridges == 255] = 1
    skeletonized_ridges = skeletonize(thresholded_ridges)

    return np.expand_dims(skeletonized_ridges, axis=-1), np.expand_dims(upsampled_peaks, axis=-1)


def compute_sketches(paths, thread_idx):
    height_maps = []
    sketch_maps = []
    i = 0
    start = time.time()
    for filename in paths:
        file_path = str(filename)
        file_id = file_path.split(os.sep)[-1]
        detailed_data = gr.from_file(os.path.join(config.PATH_TO_DATA, file_id))
        data = gr.from_file(str(filename))
        if data.mean() < 5:
            continue
        ridges, peaks = compute_ridges(filename)
        rivers, basins = compute_rivers(filename)

        height_map = np.array(detailed_data.raster, dtype=np.float32)
        height_map = np.expand_dims(height_map, axis=-1)
        height_map = (height_map - np.amin(height_map)) / \
                     (np.amax(height_map) - np.amin(height_map))
        height_map = height_map * 2 - 1

        sketch_map = np.stack((ridges, rivers, peaks, basins), axis=2)
        sketch_map = np.squeeze(sketch_map, axis=-1)

        height_maps.append(height_map)
        sketch_maps.append(sketch_map)

        if i % 500 == 0:
            print(f'Thread: {thread_idx} {i}')
        i += 1
    training_output = np.array(height_maps, dtype=np.float32)
    training_input = np.array(sketch_maps, dtype=np.float32)

    training_input_rgb = []

    for i in range(training_input.shape[0]):
        x_pic = training_input[i]
        ridges = np.zeros((config.WIDTH, config.HEIGHT, 3), dtype=np.float32)
        rivers = np.zeros((config.WIDTH, config.HEIGHT, 3), dtype=np.float32)
        peaks = np.zeros((config.WIDTH, config.HEIGHT, 3), dtype=np.float32)

        ridges[x_pic[..., 0] != 0] = [1, 0, 0]
        rivers[x_pic[..., 1] != 0] = [0, 0, 1]
        peaks[x_pic[..., 2] != 0] = [0, 1, 0]

        x_pic_rgb = np.ones((config.WIDTH, config.HEIGHT, 3), dtype=np.float32)
        x_pic_rgb[..., 0] = ridges[..., 0]
        x_pic_rgb[..., 1] = peaks[..., 1]
        x_pic_rgb[..., 2] = rivers[..., 2]

        mask = np.logical_and(x_pic_rgb[..., 0] == 0,
                              np.logical_and(x_pic_rgb[..., 1] == 0, x_pic_rgb[..., 2] == 0))
        x_pic_rgb[mask] = [1, 1, 1]

        mask = np.logical_and(ridges[..., 0] != 0, peaks[..., 1] != 0)
        x_pic_rgb[mask] = [0, 1, 0]

        training_input_rgb.append(x_pic_rgb)

    np.savez_compressed(
        '../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY + f'/sketches/{thread_idx}.npz',
        x=training_input_rgb, y=training_output)

    end = time.time()
    print(f'{thread_idx} {end - start}')


def compute_sketches_all():
    data = 'sketches'
    paths = list(Path(config.PATH_TO_DOWNSAMPLED_DATA).glob('**/*.tif'))
    count_elements = len([None for _ in os.listdir(config.PATH_TO_DOWNSAMPLED_DATA)])
    per_thread = int(count_elements / config.THREADS)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(config.THREADS - 1):
            start = per_thread * i
            end = per_thread * (i + 1)
            future = executor.submit(compute_sketches, paths[start:end], i)
            futures.append(future)

        future = executor.submit(compute_sketches, paths[(config.THREADS - 1) * per_thread:count_elements],
                                 config.THREADS - 1)
        futures.append(future)

        # Wait for all processes to finish
        for future in futures:
            future.result()

    print('sketches calculated')
    thread_results_paths = f'../{config.PATH_TO_TRAINING_DATA}{config.COUNTRY}/{data}/*.npz'
    thread_results_files = glob.glob(thread_results_paths)
    if len(thread_results_files) <= 0:
        return
    if thread_results_files[len(thread_results_files) - 1].split('\\')[1] == 'unfiltered.npz':
        # old result file
        thread_results_files.remove(thread_results_files[len(thread_results_files) - 1])
    thread_data = np.load(thread_results_files.pop())

    result_sketches = thread_data['x']
    result_height = thread_data['y']

    for i in range(len(thread_results_files)):
        thread_data = np.load(thread_results_files[i])

        thread_data_sketches = thread_data['x']
        thread_data_height = thread_data['y']

        result_sketches = np.concatenate((result_sketches, thread_data_sketches), axis=0)
        result_height = np.concatenate((result_height, thread_data_height), axis=0)

    np.savez_compressed('../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY + '/' + data + '/unfiltered.npz',
                        x=result_sketches, y=result_height)


def compute_binary_map(im):
    threshold = np.percentile(im, config.PERCENTILE)
    target_image = np.zeros(im.shape, dtype=np.float32)
    target_image[im >= threshold] = 1

    return target_image, threshold


def compute_levelset(paths, thread_idx):
    height_maps = []
    levelset_maps = []
    threshold_maps = []
    i = 0
    max_height = np.finfo(np.float32).min
    min_height = np.finfo(np.float32).max
    for filename in paths:
        file_path = str(filename)
        file_id = file_path.split(os.sep)[-1]
        detailed_data = gr.from_file(os.path.join(config.PATH_TO_DATA, file_id))
        data = gr.from_file(str(filename))

        bluerred_image = cv2.GaussianBlur(np.array(data.raster, dtype=np.float32), (5, 5), 0)
        levelset, threshold = compute_binary_map(bluerred_image)

        levelset_upsampled = cv2.pyrUp(np.array(levelset, dtype=np.float32), dstsize=(config.WIDTH, config.HEIGHT))

        levelset_map = np.array(levelset_upsampled, dtype=np.float32)
        levelset_map = np.expand_dims(levelset_map, axis=-1)

        height_map = np.array(detailed_data.raster, dtype=np.float32)
        height_map = np.expand_dims(height_map, axis=-1)

        if max_height < np.amax(height_map):
            max_height = np.amax(height_map)

        if min_height > np.amin(height_map):
            min_height = np.amin(height_map)

        height_map = (height_map - np.amin(height_map)) / \
                     (np.amax(height_map) - np.amin(height_map))

        height_maps.append(height_map)
        levelset_maps.append(levelset_map)
        threshold_maps.append(threshold)

        if i % 500 == 0:
            print(f'Thread: {thread_idx} {i}')
        i += 1

    training_input = np.array(levelset_maps, dtype=np.float32)
    training_output = np.array(height_maps, dtype=np.float32)

    np.savez_compressed(f'../{config.PATH_TO_TRAINING_DATA}{config.COUNTRY}/levelset/{thread_idx}.npz',
        x=training_input, y=training_output, threshold=threshold_maps, max=max_height, min=min_height)


def compute_levelset_all():
    data = 'levelset'
    paths = list(Path(config.PATH_TO_DOWNSAMPLED_DATA).glob('**/*.tif'))
    count_elements = len([None for _ in os.listdir(config.PATH_TO_DOWNSAMPLED_DATA)])
    per_thread = int(count_elements / config.THREADS)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(config.THREADS - 1):
            start = per_thread * i
            end = per_thread * (i + 1)
            future = executor.submit(compute_levelset, paths[start:end], i)
            futures.append(future)

        future = executor.submit(compute_levelset, paths[(config.THREADS - 1) * per_thread:count_elements],
                                 config.THREADS - 1)
        futures.append(future)

        # Wait for all processes to finish
        for future in futures:
            future.result()

    print('levelset calculated')

    thread_results_paths = f'../{config.PATH_TO_TRAINING_DATA}{config.COUNTRY}/{data}/*.npz'
    thread_results_files = glob.glob(thread_results_paths)
    if len(thread_results_files) <= 0:
        return
    if thread_results_files[len(thread_results_files) - 1].split('\\')[1] == 'unfiltered.npz':
        # old result file
        thread_results_files.remove(thread_results_files[len(thread_results_files) - 1])
    thread_data = np.load(thread_results_files.pop())

    result_levelset = thread_data['x']
    result_height = thread_data['y']
    result_threshold = thread_data['threshold']
    result_max = thread_data['max']
    result_min = thread_data['min']

    for i in range(len(thread_results_files)):
        thread_data = np.load(thread_results_files[i])
        thread_data_levelset = thread_data['x']
        thread_data_height = thread_data['y']
        thread_data_max = thread_data['max']
        thread_data_min = thread_data['min']
        thread_data_threshold = thread_data['threshold']
        result_max = max(result_max, thread_data_max)
        result_min = max(result_min, thread_data_min)

        result_height = np.concatenate((result_height, thread_data_height), axis=0)
        result_levelset = np.concatenate((result_levelset, thread_data_levelset), axis=0)
        result_threshold = np.concatenate((result_threshold, thread_data_threshold), axis=0)

    np.savez_compressed(f'../{config.PATH_TO_TRAINING_DATA}{config.COUNTRY}/{data}/unfiltered.npz',
                        x=result_levelset, y=result_height, threshold=result_threshold, max=result_max, min=result_min)


def filter_levelset_data_levelset_height():
    data = np.load(f'../{config.PATH_TO_TRAINING_DATA}{config.COUNTRY}/levelset/unfiltered.npz')

    levelset_maps = data['x']
    height_maps = data['y']
    threshold_all = data['threshold']

    height_maps_filtered = []
    levelset_maps_filtered = []

    j = 0
    lower_threshold = np.percentile(threshold_all, 30)
    upper_threshold = np.percentile(threshold_all, 70)
    for i in range(len(height_maps)):
        height_map = height_maps[i]
        threshold = threshold_all[i]
        levelset_map = levelset_maps[i]

        if (threshold >= lower_threshold) and (threshold <= upper_threshold):
            levelset_map = (levelset_map - np.amin(levelset_map)) / \
                         (np.amax(levelset_map) - np.amin(levelset_map))

            height_maps_filtered.append(height_map)
            levelset_maps_filtered.append(levelset_map)
            j += 1

    print(j)

    training_input = np.array(levelset_maps_filtered, dtype=np.float32)
    training_output = np.array(height_maps_filtered, dtype=np.float32)
    np.savez_compressed('../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY + '/levelset/levelset.npz',
                        x=training_input, y=training_output)


def compute_eraser(paths, thread_idx):
    height_maps = []
    eraser_maps = []
    state = np.random.RandomState(thread_idx)
    i = 0
    for filename in paths:
        file_path = str(filename)
        file_id = file_path.split(os.sep)[-1]
        detailed_data = gr.from_file(os.path.join(config.PATH_TO_DATA, file_id))

        data = np.float32(detailed_data.raster)
        circles = create_circles(state, data)
        eraser_map = compute_erased_im(data, circles)

        eraser_map = np.array(eraser_map, dtype=np.float32)
        eraser_map = np.expand_dims(eraser_map, axis=-1)
        eraser_map = (eraser_map - np.amin(eraser_map)) / \
                     (np.amax(eraser_map) - np.amin(eraser_map))

        height_map = np.array(detailed_data.raster, dtype=np.float32)
        height_map = np.expand_dims(height_map, axis=-1)
        height_map = (height_map - np.amin(height_map)) / \
                     (np.amax(height_map) - np.amin(height_map))

        height_maps.append(height_map)
        eraser_maps.append(eraser_map)

        if i % 500 == 0:
            print(f'Thread: {thread_idx} {i}')
        i += 1

    training_input = np.array(eraser_maps, dtype=np.float32)
    training_output = np.array(height_maps, dtype=np.float32)

    np.savez_compressed(f'../{config.PATH_TO_TRAINING_DATA}{config.COUNTRY}/eraser/{thread_idx}.npz',
                        x=training_input, y=training_output)


def compute_eraser_all():
    data = 'eraser'
    paths = list(Path(config.PATH_TO_DOWNSAMPLED_DATA).glob('**/*.tif'))
    count_elements = len([None for _ in os.listdir(config.PATH_TO_DOWNSAMPLED_DATA)])
    per_thread = int(count_elements / config.THREADS)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(config.THREADS - 1):
            start = per_thread * i
            end = per_thread * (i + 1)
            future = executor.submit(compute_eraser, paths[start:end], i)
            futures.append(future)

        future = executor.submit(compute_eraser, paths[(config.THREADS-1) * per_thread:count_elements], config.THREADS-1)
        futures.append(future)

        for future in futures:
            future.result()

    print('eraser calculated')

    thread_results_paths = f'../{config.PATH_TO_TRAINING_DATA}{config.COUNTRY}/{data}/*.npz'
    thread_results_files = glob.glob(thread_results_paths)
    if len(thread_results_files) <= 0:
        return
    if thread_results_files[len(thread_results_files) - 1].split('\\')[1] == 'unfiltered.npz':
        # old result file
        thread_results_files.remove(thread_results_files[len(thread_results_files) - 1])
    thread_data = np.load(thread_results_files.pop())

    result_eraser = thread_data['x']
    result_height = thread_data['y']

    for i in range(len(thread_results_files)):
        thread_data = np.load(thread_results_files[i])

        thread_data_vegetation = thread_data['x']
        thread_data_height = thread_data['y']

        result_height = np.concatenate((result_height, thread_data_height), axis=0)
        result_eraser = np.concatenate((result_eraser, thread_data_vegetation), axis=0)

    np.savez_compressed(f'../{config.PATH_TO_TRAINING_DATA}{config.COUNTRY}/{data}/unfiltered.npz',
                        x=result_eraser, y=result_height)


if __name__ == '__main__':
    extract_patches_from_raster()

    #compute_sketches_all()

    #compute_levelset_all()
    #filter_levelset_data_levelset_height()

    #compute_eraser_all()
