import os

import sys
import time
import numpy as np
from skimage.util import view_as_blocks
from pathlib import Path
import warnings

from utils import has_more_than_x_vegetation_percent_coverage, calculate_tertiary_map_vegetation_sketches

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import georasters as gr
import cv2
import concurrent.futures
import glob

import config
from utils import compute_erased_im, create_circles


def extract_patches_from_raster():
    count = 0
    start = time.time()
    for raster_file in Path(config.PATH_TO_INPUT).glob('**/*.tif'):
        data = gr.from_file(str(raster_file))

        if 0:
            # resize an image s.t. blocks with config.width x config.height can be extracted
            data = data.resize((int(data.shape[0] / config.WIDTH) * config.WIDTH,
                                (int(data.shape[1] / config.HEIGHT) * config.HEIGHT)))
            raster_blocks = view_as_blocks(data.raster, (config.WIDTH, config.HEIGHT))
        else:
            # Resize images s.t. the 3x3x foot images can later be scaled up to 7x7 m resolution per pixel
            # 23 foot approx 7m
            data = data.resize((int(data.shape[0] / 2048) * 2048, (int(data.shape[1] / 2048) * 2048)))
            raster_blocks = view_as_blocks(data.raster, (2048, 2048))  # (config.WIDTH, config.HEIGHT))

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


def transform_to_7m_resolution(paths):
    # transform maps to 7m resolution
    maps = []
    for path in paths:
        detailed_data = gr.from_file(path)
        map = cv2.resize(np.array(detailed_data.raster, dtype=np.float32),
                         (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)

        if np.amin(map) < -100000:
            # values below -100000 signals image with pixels outside
            continue

        map = np.array(map, dtype=np.float32)
        maps.append(map)
    return maps


def compute_sketch_to_vegetation(paths_height, thread_idx, global_scale=False, eraser=False, veg_coverage_filter=False):
    height_maps = []
    vegetation_maps = []
    input_vegetation_maps = []
    h_max = sys.float_info.min
    h_min = sys.float_info.max
    count = 0
    start = time.time()
    for path in paths_height:
        if count % 50 == 0:
            if thread_idx == 0:
                end = time.time()
                print(f'Thread: {thread_idx} {count} {end - start}')
                start = time.time()
            elif count % 150 == 0:
                print(f'Thread: {thread_idx} {count}')
        file_path = str(path)
        file_id = file_path.split(os.sep)[-1]

        detailed_data_height = gr.from_file(path)

        data_veg_in = gr.from_file(os.path.join('../data/usa_vegetation/data_vegetation_intensity/', file_id))
        data_veg_in_downsampled = gr.from_file \
            (os.path.join('../data/usa_vegetation/transformed_input_vegetation_intensity/', file_id))

        height_map = np.array(detailed_data_height.raster, dtype=np.float32)
        vegetation_in_map = np.array(data_veg_in.raster, dtype=np.float32)
        vegetation_in_downsampled_map = np.array(data_veg_in_downsampled.raster, dtype=np.float32)

        if abs(np.amin(height_map) + 9999.0) < 0.001 or abs(np.amin(vegetation_in_map) + 9999.0) < 0.001:
            continue

        # calculate input_vegetation_map with operations close from downsampled vegetation intensity
        kernel_erode_and_dilate = np.ones((5, 5), dtype=np.uint8)
        dilate = cv2.dilate(vegetation_in_downsampled_map, kernel_erode_and_dilate, iterations=2)

        close_op = cv2.erode(dilate, kernel_erode_and_dilate, iterations=2)

        close_op = cv2.GaussianBlur(np.array(close_op, dtype=np.float32), (5, 5), 2)

        close_upsampled = cv2.pyrUp(np.array(close_op, dtype=np.float32), dstsize=(config.WIDTH, config.HEIGHT))
        close_map = np.array(close_upsampled, dtype=np.float32)
        close_map = np.expand_dims(close_map, axis=-1)
        close_map = (close_map - np.amin(close_map)) / (np.amax(close_map) - np.amin(close_map))

        if veg_coverage_filter:
            if has_more_than_x_vegetation_percent_coverage(close_map, 80):
                continue
            else:
                close_map = calculate_tertiary_map_vegetation_sketches(close_map)

        input_vegetation_maps.append(close_map)

        vegetation_in_map = np.array(vegetation_in_map, dtype=np.float32)
        vegetation_in_map = np.expand_dims(vegetation_in_map, axis=-1)
        vegetation_in_map = (vegetation_in_map - np.amin(vegetation_in_map)) / (
                np.amax(vegetation_in_map) - np.amin(vegetation_in_map))

        if global_scale:
            h_max = max(np.amax(height_map), h_max)
            h_min = min(np.amin(height_map), h_min)

        else:
            height_map = np.expand_dims(height_map, axis=-1)
            height_map = (height_map - np.amin(height_map)) / \
                         (np.amax(height_map) - np.amin(height_map))

        height_maps.append(height_map)
        vegetation_maps.append(vegetation_in_map)
        count += 1
        import matplotlib.pyplot as plt
        plt.imsave(f'../{config.PATH_TO_VEGETATION_OUTPUT}generated/asdasdasd.png', np.squeeze(height_map, axis=-1), cmap='gray')
        plt.imsave(f'../{config.PATH_TO_VEGETATION_OUTPUT}generated/asdasdasdasd.png', np.squeeze(vegetation_in_map, axis=-1), cmap='gray')
        return
    mapped_height_maps = []
    if global_scale:
        for height_map in height_maps:
            height_map = np.expand_dims(height_map, axis=-1)
            height_map = (height_map - h_min) / (h_max - h_min)
            mapped_height_maps.append(height_map)
    else:
        mapped_height_maps = height_maps

    training_input_height = np.array(mapped_height_maps, dtype=np.float32)
    training_input_vegetation = np.array(input_vegetation_maps, dtype=np.float32)
    training_output = np.array(vegetation_maps, dtype=np.float32)

    np.savez_compressed(
        '../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY + f'/vegetation/{thread_idx}.npz',
        x=training_input_height, y=training_output, z=training_input_vegetation)


def compute_eraser_vegetation(paths_height, thread_idx, global_scale=False, eraser=False, veg_coverage_filter=False):
    height_maps = []
    vegetation_maps = []
    input_vegetation_maps = []
    count = 0
    h_max = sys.float_info.min
    h_min = sys.float_info.max
    start = time.time()
    state = np.random.RandomState(thread_idx)
    for path in paths_height:
        if count % 50 == 0:
            if thread_idx == 0:
                end = time.time()
                print(f'Thread: {thread_idx} {count} {end - start}')
                start = time.time()
            elif count % 150 == 0:
                print(f'Thread: {thread_idx} {count}')
        file_path = str(path)
        file_id = file_path.split(os.sep)[-1]

        detailed_data_height = gr.from_file(path)

        data_veg_in = gr.from_file(os.path.join('../data/usa_vegetation/data_vegetation_intensity/', file_id))

        height_map = np.array(detailed_data_height.raster, dtype=np.float32)
        vegetation_in_map = np.array(data_veg_in.raster, dtype=np.float32)

        if abs(np.amin(height_map) + 9999.0) < 0.001 or abs(np.amin(vegetation_in_map) + 9999.0) < 0.001:
            continue

        # create erased vegetation map
        circles = create_circles(state, vegetation_in_map)
        erased_veg_map = compute_erased_im(vegetation_in_map, circles)
        input_vegetation_maps.append(erased_veg_map)

        vegetation_in_map = np.array(vegetation_in_map, dtype=np.float32)
        vegetation_in_map = np.expand_dims(vegetation_in_map, axis=-1)
        vegetation_in_map = ((vegetation_in_map - np.amin(vegetation_in_map)) /
                             (np.amax(vegetation_in_map) - np.amin(vegetation_in_map)))

        if global_scale:
            h_max = max(np.amax(height_map), h_max)
            h_min = min(np.amin(height_map), h_min)

        else:
            height_map = np.expand_dims(height_map, axis=-1)
            height_map = (height_map - np.amin(height_map)) / \
                         (np.amax(height_map) - np.amin(height_map))

        height_maps.append(height_map)
        vegetation_maps.append(vegetation_in_map)
        count += 1

    mapped_height_maps = []
    if global_scale:
        for height_map in height_maps:
            height_map = np.expand_dims(height_map, axis=-1)
            height_map = (height_map - h_min) / (h_max - h_min)
            mapped_height_maps.append(height_map)
    else:
        mapped_height_maps = height_maps

    training_input_height = np.array(mapped_height_maps, dtype=np.float32)
    training_input_vegetation = np.array(input_vegetation_maps, dtype=np.float32)
    training_output = np.array(vegetation_maps, dtype=np.float32)

    np.savez_compressed('../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY + f'/vegetation/{thread_idx}.npz',
                        x=training_input_height, y=training_output, z=training_input_vegetation)


def compute_height_to_vegetation(paths_height, thread_idx, global_scale=False, eraser=False,veg_coverage_filter=False):
    height_maps = []
    vegetation_maps = []
    input_vegetation_maps = []
    h_max = sys.float_info.min
    h_min = sys.float_info.max
    count = 0
    start = time.time()
    for path in paths_height:
        if count % 50 == 0:
            if thread_idx == 0:
                end = time.time()
                print(f'Thread: {thread_idx} {count} {end - start}')
                start = time.time()
            elif count % 150 == 0:
                print(f'Thread: {thread_idx} {count}')
        file_path = str(path)
        file_id = file_path.split(os.sep)[-1]

        detailed_data_height = gr.from_file(path)

        data_veg_in = gr.from_file(os.path.join('../data/usa_vegetation/data_vegetation_intensity/', file_id))

        height_map = np.array(detailed_data_height.raster, dtype=np.float32)
        vegetation_in_map = np.array(data_veg_in.raster, dtype=np.float32)

        if abs(np.amin(height_map) + 9999.0) < 0.001 or abs(np.amin(vegetation_in_map) + 9999.0) < 0.001:
            continue

        vegetation_in_map = np.expand_dims(vegetation_in_map, axis=-1)
        vegetation_in_map = (vegetation_in_map - np.amin(vegetation_in_map)) / (
                np.amax(vegetation_in_map) - np.amin(vegetation_in_map))

        if global_scale:
            h_max = max(np.amax(height_map), h_max)
            h_min = min(np.amin(height_map), h_min)

        else:
            height_map = np.expand_dims(height_map, axis=-1)
            height_map = (height_map - np.amin(height_map)) / (np.amax(height_map) - np.amin(height_map))

        height_maps.append(height_map)
        vegetation_maps.append(vegetation_in_map)
        count += 1

    mapped_height_maps = []
    if global_scale:
        for height_map in height_maps:
            height_map = np.expand_dims(height_map, axis=-1)
            height_map = (height_map - h_min) / (h_max - h_min)
            mapped_height_maps.append(height_map)
    else:
        mapped_height_maps = height_maps

    training_input_height = np.array(mapped_height_maps, dtype=np.float32)
    training_input_vegetation = np.array(input_vegetation_maps, dtype=np.float32)
    training_output = np.array(vegetation_maps, dtype=np.float32)

    np.savez_compressed(
        '../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY + f'/vegetation/{thread_idx}.npz',
        x=training_input_height, y=training_output, z=training_input_vegetation)


def compute_vegetation_all(func, global_scale=False, eraser=False, veg_coverage_filter=False):
    data = 'vegetation'
    file_pattern_height = str('../data/usa_vegetation/data_height/*.tif')

    path_height = glob.glob(file_pattern_height)

    count_elements = len([None for _ in os.listdir(config.PATH_TO_DATA)])
    per_thread = int(count_elements / config.THREADS)

    s_time = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(config.THREADS - 1):
            start = per_thread * i
            end = per_thread * (i + 1)
            future = executor.submit(func, path_height[start:end], i,
                                     global_scale, eraser, veg_coverage_filter)
            futures.append(future)

        future = executor.submit(func, path_height[(config.THREADS - 1) * per_thread:count_elements],
                                 config.THREADS - 1, global_scale, eraser, veg_coverage_filter)
        futures.append(future)

        for future in futures:
            future.result()

    e_time = time.time()
    print(f'time count: {e_time - s_time}')

    print('vegetation calculated')

    # merge the thread results together
    thread_results_paths = '../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY + '/' + data + f'/*.npz'
    thread_results_files = glob.glob(thread_results_paths)
    if len(thread_results_files) <= 0:
        return
    if thread_results_files[len(thread_results_files) - 1].split('\\')[1] == 'unfiltered.npz':
        thread_results_files.remove(thread_results_files[len(thread_results_files) - 1])
    thread_data = np.load(thread_results_files.pop())

    result_height = thread_data['x']
    result_output_vegetation = thread_data['y']
    result_input_vegetation = thread_data['z']
    result_erode_maps = thread_data['erode']
    result_dilate_maps = thread_data['dilate']

    for i in range(len(thread_results_files)):
        thread_data = np.load(thread_results_files[i])
        thread_data_height = thread_data['x']
        thread_data_output_vegetation = thread_data['y']
        thread_data_input_vegetation = thread_data['z']
        thread_data_erode = thread_data['erode']
        thread_data_dilate = thread_data['dilate']

        result_height = np.concatenate((result_height, thread_data_height), axis=0)
        result_output_vegetation = np.concatenate((result_output_vegetation, thread_data_output_vegetation), axis=0)
        result_input_vegetation = np.concatenate((result_input_vegetation, thread_data_input_vegetation), axis=0)
        result_erode_maps = np.concatenate((result_erode_maps, thread_data_erode), axis=0)
        result_dilate_maps = np.concatenate((result_dilate_maps, thread_data_dilate), axis=0)

    np.savez_compressed('../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY + '/' + data + '/unfiltered.npz',
                        x=result_height, y=result_output_vegetation, z=result_input_vegetation, erode=result_erode_maps,
                        dilate=result_dilate_maps)


if __name__ == '__main__':
    compute_vegetation_all(compute_height_to_vegetation, global_scale=False, eraser=False, veg_coverage_filter=False)
    #compute_vegetation_all(compute_sketch_to_vegetation, global_scale=False, eraser=False, veg_coverage_filter=True)
    #compute_vegetation_all(compute_eraser_vegetation, global_scale=False, eraser=False, veg_coverage_filter=True)