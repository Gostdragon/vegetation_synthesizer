import sys

sys.path.append('src/')
import argparse

import tf_pix2pix.sketch_to_terrain_p2p as s_p2p
import tf_pix2pix.levelset_to_terrain_p2p as l_p2p
import tf_pix2pix.eraser_to_terrain_p2p as e_p2p
import tf_pix2pix.height_to_vegetation_minecraft as v_m_p2p
import tf_pix2pix.sketch_to_vegetation as s_v
import tf_pix2pix.height_to_vegetation as v_usa_p2p
import tf_pix2pix.eraser_for_vegetation as e_v

from file_handler import *
from utils import *
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--png16bits", dest="png16bits", action="store_true",
                        help="use png 16 bits images encoder and decoders")
    parser.set_defaults(png16bits=False)

    args = parser.parse_args()

    #plt.imsave(config.PATH_TO_SKETCHES_OUTPUT + 'own_sketches/sketch.png', np.ones((256, 256), dtype=np.float32), cmap='gray')

    #save_sketches_picture_from_training_data()
    #s_p2p.train_gan(args)
    #s_p2p.create_images(args, 100)
    #s_p2p.create_terrain_from_sketch(args, config.PATH_TO_SKETCHES_OUTPUT + 'own_sketches/sketches/', cut_sketch=False)

    #save_levelset_pictures_from_training_data()
    #l_p2p.train_gan(args)
    #l_p2p.create_images(args, 17)

    #save_eraser_pictures_from_training_data()
    #e_p2p.train_gan(args)
    #e_p2p.create_images(args, 1)

    #save_vegetation_pictures_from_training_data()
    # classify_realistic_minecraft_train_images()
    #v_m_p2p.train_gan(args)
    #v_m_p2p.create_images(args, 10)
    #v_m_p2p.create_vegetation_from_heightmap(args, config.PATH_TO_VEGETATION_OUTPUT + 'own_heightmaps/heightmaps')

    #save_vegetation_pictures_from_training_data()
    #v_usa_p2p.train_gan(args)
    #v_usa_p2p.create_images(args, 100)
    #v_usa_p2p.create_vegetation_from_heightmap(args, config.PATH_TO_VEGETATION_OUTPUT + 'own_heightmaps/heightmaps')


    #create_vertical_slices(args)
    #blurr_generated_images(args, config.PATH_TO_SKETCHES_OUTPUT + '/generated/', config.PATH_TO_SKETCHES_OUTPUT + '/generated/', False)
    #utils.convert_x_to_x_half_resolution()
    #utils.convert_x_half_to_x_resolution()


if __name__ == '__main__':
    main()
