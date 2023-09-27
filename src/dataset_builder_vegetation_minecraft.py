import math
import time
import numpy as np
from pathlib import Path
import warnings
import concurrent.futures

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import anvil
import config


def find_first_non_air_block(chunk, x, z, start=319, step_size=20):
    # returns height of first non-air block
    y = min(start, 319)

    while y > 0:
        block = chunk.get_block(x, y, z)
        if block.id not in config.NON_SOLID_BLOCKS:
            for i in range(y + step_size - 1, y, -1):
                block = chunk.get_block(x, i, z)
                if block.id not in config.NON_SOLID_BLOCKS:
                    y = i
                    break
            return y
            # if blo
            # ck.id in config.VEGETATION_BLOCKS:
            #     veg_hit = True
            # else:
            #     height_map[x, z] = y
            #     # veg_map[x, z] = y
            #     air_block = False
        y -= step_size
    return 0


def get_ground_and_wood_height(c, x, z, height):
    # returns height of first non air or non vegetation block and height of first vegetation block

    block = c.get_block(x, height, z)
    hit_wood = False
    wood_height = 0
    while block.id.find('leaves') > -1 or block.id.find('log') > -1 or block.id in config.NON_SOLID_BLOCKS: #block.id in config.VEGETATION_BLOCKS :
        if block.id.find('log') > -1 and not hit_wood: #block.id in config.WOOD_BLOCKS and not hit_wood:
            wood_height = height
            hit_wood = True

        height -= 1
        block = c.get_block(x, height, z)

    return height, wood_height


def get_water_height(c, x, z, height):
    # returns height of first water block with a non water block above

    block = c.get_block(x, height, z)

    while height <= 319:
        if block.id not in config.WATER_BLOCKS:
            break
        else:
            height += 1
            block = c.get_block(x, height, z)

    return height - 1


def get_height_maps(c):
    height_map = np.zeros((config.MINECRAFT_CHUNK_SIZE, config.MINECRAFT_CHUNK_SIZE, 1), dtype=np.uint32)
    veg_map = np.zeros((config.MINECRAFT_CHUNK_SIZE, config.MINECRAFT_CHUNK_SIZE, 1), dtype=np.uint32)
    water_map = np.zeros((config.MINECRAFT_CHUNK_SIZE, config.MINECRAFT_CHUNK_SIZE, 1), dtype=np.uint32)
    for x in range(config.MINECRAFT_CHUNK_SIZE):
        for z in range(config.MINECRAFT_CHUNK_SIZE):
            y = 319
            veg_height = 0
            water_height = 0

            height = find_first_non_air_block(c, x, z, start=y)
            block = c.get_block(x, height, z)

            if block.id.find('log') > -1 or block.id.find('leaves') > -1:#in config.VEGETATION_BLOCKS:
                height, veg_height = get_ground_and_wood_height(c, x, z, height)

            if height + 1 < 319:
                block_above = c.get_block(x, height + 1, z)
                if block_above.id in config.WATER_BLOCKS:
                    water_height = get_water_height(c, x, z, height + 1)

            height_map[x, z] = height
            veg_map[x, z] = veg_height
            water_map[x, z] = water_height

    return height_map, veg_map, water_map


def stitch_chunks_together(thread_idx):
    data = np.load('../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY + f'/vegetation/{thread_idx}.npz')
    height_maps = data['x']
    veg_maps = data['y']
    water_maps = data['z']
    joint_height_maps = []
    joint_veg_maps = []
    joint_water_maps = []
    for maps in range(len(height_maps)):
        for idx_image in range(4):
            big_joint_height_map = None
            big_joint_veg_map = None
            big_joint_water_map = None
            for i in range(0, len(height_maps[maps][idx_image]), config.MINECRAFT_CHUNK_SIZE):
                # chunks are serialized -> stitch them together
                # ech train image is 256x256, each chunk is 16x16
                joint_height_map = height_maps[maps][idx_image][i]
                joint_height_map = np.squeeze(joint_height_map, axis=-1)
                joint_veg_map = veg_maps[maps][idx_image][i]
                joint_veg_map = np.squeeze(joint_veg_map, axis=-1)
                joint_water_map = water_maps[maps][idx_image][i]
                joint_water_map = np.squeeze(joint_water_map, axis=-1)
                for j in range(1, config.MINECRAFT_CHUNK_SIZE):
                    joint_height_map = np.concatenate(
                        [joint_height_map, np.squeeze(height_maps[maps][idx_image][i + j], axis=-1)], axis=1)
                    joint_veg_map = np.concatenate([joint_veg_map, np.squeeze(veg_maps[maps][idx_image][i + j], axis=-1)],
                                                   axis=1)
                    joint_water_map = np.concatenate(
                        [joint_water_map, np.squeeze(water_maps[maps][idx_image][i + j], axis=-1)], axis=1)
                if i == 0:
                    big_joint_height_map = joint_height_map
                    big_joint_veg_map = joint_veg_map
                    big_joint_water_map = joint_water_map
                else:
                    big_joint_height_map = np.concatenate([big_joint_height_map, joint_height_map], axis=0)
                    big_joint_veg_map = np.concatenate([big_joint_veg_map, joint_veg_map], axis=0)
                    big_joint_water_map = np.concatenate([big_joint_water_map, joint_water_map], axis=0)

            joint_height_maps.append(big_joint_height_map)
            joint_veg_maps.append(big_joint_veg_map)
            joint_water_maps.append(big_joint_water_map)

    return joint_height_maps, joint_veg_maps, joint_water_maps


def compute_vegetation(paths, thread_idx):
    i = 0
    height_maps = []
    veg_maps = []
    water_maps = []
    for path in paths:
        # each region consists of 32x32 chunks
        # with a chunk size of 16, each region consists of 4 train images
        print(f'{thread_idx}: file {i} start')
        region = anvil.Region.from_file(str(path))
        height_map = [[] for _ in range(4)]
        veg_map = [[] for _ in range(4)]
        water_map = [[] for _ in range(4)]
        start = time.time()
        for x in range(32):
            for y in range(32):
                c = anvil.Chunk.from_region(region, x, y)

                cal_height_map, cal_veg_map, cal_water_map = get_height_maps(c)

                height_map[math.floor(x / 16) * 2 + math.floor(y / 16)].append(cal_height_map)
                veg_map[math.floor(x / 16) * 2 + math.floor(y / 16)].append(cal_veg_map)
                water_map[math.floor(x / 16) * 2 + math.floor(y / 16)].append(cal_water_map)

        height_map = (height_map - np.amin(height_map)) / \
                               max(1, (np.amax(height_map) - np.nanmin(np.array([0, np.amin(height_map)]))))
        height_maps.append(height_map)

        veg_map = (veg_map - np.amin(veg_map)) / \
                            max(1, (np.amax(veg_map) - np.nanmin(np.array([0, np.amin(veg_map)]))))
        veg_maps.append(veg_map)

        water_map = (water_map - np.amin(water_map)) / \
                              max(1, (np.amax(water_map) - np.nanmin(np.array([0, np.amin(water_map)]))))
        water_maps.append(water_map)

        end = time.time()
        if thread_idx == 0:
            print(f'{thread_idx}: file {i} end {end - start}')
        else:
            print(f'{thread_idx}: file {i} end')
        i += 1
    np.savez_compressed('../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY +
                       f'/vegetation/{thread_idx}.npz', x=height_maps, y=veg_maps, z=water_maps)

    joint_height_maps, joint_veg_maps, joint_water_maps = stitch_chunks_together(thread_idx)

    np.savez_compressed('../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY +
                        f'/vegetation/{thread_idx}_stitched.npz', x=joint_height_maps, y=joint_veg_maps,
                        z=joint_water_maps)


def foo(paths, idx):
    for i in paths:
        im = plt.imread(i)
        w = im.shape[1] // 2

        height_m = im[:, :w, 0:3]
        veg = im[:, w:, 1]
        water = im[:, w:, 2]

        width = veg.shape[0]
        height = veg.shape[1]
        next_res = veg.copy()
        cur_res = veg.copy()
        for _ in range(4):
            for x in range(width):
                for y in range(height):
                    for k in range(-4, 5):
                        for j in range(-4, 5):
                            if width > x + k >= 0 and 0 <= y + j < height:
                                if k == 0 and j == 0:
                                    continue
                                div = 1
                                mul = 1
                                dis = math.sqrt(k * k + j * j)
                                if dis <= 3:
                                    div = 2
                                    mul = 1
                                elif dis <= 4:
                                    div = 1.5
                                    mul = 0.5
                                # elif dis <= 3:
                                #     div = 1.25
                                #     mul = 0.25
                                # elif dis <= 4:
                                #     div = 1.125
                                #     mul = 0.125

                                next_res[x + k][y + j] += (cur_res[x + k][y + j] + mul * cur_res[x][y]) / div
            cur_res = next_res.copy()

        cur_res = (cur_res - np.amin(cur_res)) / \
                    (np.amax(cur_res) - np.amin(cur_res))

        res_3dim = np.zeros((veg.shape[0], veg.shape[1], 3), dtype=np.float32)
        res_3dim[..., 1] = cur_res
        res_3dim[..., 2] = water
        im = np.concatenate([height_m, res_3dim], axis=1)

        plt.imsave('../' + config.PATH_TO_VEGETATION_OUTPUT + 'train_minecraft_realistic/veg/asda.png', im)
        plt.imsave('../' + config.PATH_TO_VEGETATION_OUTPUT + 'train_minecraft_realistic/veg/green.png', cur_res)


def compute_vegetation_all():
    data = 'vegetation'
    paths = list(Path(config.PATH_TO_INPUT).glob('**/*.mca'))

    paths = list(Path(config.PATH_TO_VEGETATION_OUTPUT + 'train_minecraft_realistic/veg').glob('*.png'))

    count_elements = len(paths)
    per_thread = int(count_elements / config.THREADS)
    startt = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(config.THREADS - 1):
            start = per_thread * i
            end = per_thread * (i + 1)
            future = executor.submit(compute_vegetation, paths[start:end], i)
            futures.append(future)

        future = executor.submit(compute_vegetation, paths[(config.THREADS - 1) * per_thread:count_elements],
                                 config.THREADS - 1)
        futures.append(future)

        # Wait for all processes to finish
        for future in futures:
            future.result()
    endt = time.time()
    print(f'vegetation calculated {endt - startt}')

    files = list(Path(config.PATH_TO_TRAINING_DATA + config.COUNTRY + '/vegetation/').glob('**/*stitched.npz'))
    thread_data = np.load(str(files.pop()))
    result_height = thread_data['x']
    result_vegetation = thread_data['y']
    result_water = thread_data['z']
    for file in files:
        thread_data = np.load(str(file))

        thread_data_height = thread_data['x']
        thread_data_vegetation = thread_data['y']
        thread_data_water = thread_data['z']

        result_height = np.concatenate((result_height, thread_data_height), axis=0)
        result_vegetation = np.concatenate((result_vegetation, thread_data_vegetation), axis=0)
        result_water = np.concatenate((result_water, thread_data_water), axis=0)

    np.savez_compressed('../' + config.PATH_TO_TRAINING_DATA + config.COUNTRY + '/' + data + '/vegetation-water.npz',
                        x=result_height, y=result_vegetation, z=result_water)


if __name__ == '__main__':
    compute_vegetation_all()