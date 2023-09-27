# data paths
COUNTRY = 'usa'
kind = ''

PATH_TO_INPUT = '../data/' + COUNTRY + '/input' + kind + '/'
PATH_TO_DOWNSAMPLED_DATA = '../data/' + COUNTRY + '/transformed_input' + kind + '/'
PATH_TO_DATA = '../data/' + COUNTRY + '/data' + kind + '/'
PATH_TO_VEGETATION_DATA = 'data/' + COUNTRY + '/vegetation/'


PATH_TO_SKETCHES_OUTPUT = 'synthesizer/sketches/'
PATH_TO_SKETCH_MILESTONE = PATH_TO_SKETCHES_OUTPUT + 'milestone/sketch_' + COUNTRY + '/'
PATH_TO_SKETCH_MODEL_MILESTONE = PATH_TO_SKETCHES_OUTPUT + 'milestone/model_' + COUNTRY + '/'

PATH_TO_LEVELSET_OUTPUT = 'synthesizer/levelset/'
PATH_TO_LEVELSET_MILESTONE = PATH_TO_LEVELSET_OUTPUT + 'milestone/levelset_' + COUNTRY + '/'
PATH_TO_LEVELSET_MODEL_MILESTONE = PATH_TO_LEVELSET_OUTPUT + '/milestone/model_' + COUNTRY + '/'

PATH_TO_ERASER_OUTPUT = 'synthesizer/eraser/'
PATH_TO_ERASER_MILESTONE = PATH_TO_ERASER_OUTPUT + 'milestone/eraser_' + COUNTRY + '/'
PATH_TO_ERASER_MODEL_MILESTONE = PATH_TO_ERASER_OUTPUT + 'milestone/model_' + COUNTRY + '/'

PATH_TO_VEGETATION_OUTPUT = 'synthesizer/vegetation/'
PATH_TO_VEGETATION_MILESTONE = PATH_TO_VEGETATION_OUTPUT + 'milestone/vegetation_' + COUNTRY + '/'
PATH_TO_VEGETATION_MODEL_MILESTONE = PATH_TO_VEGETATION_OUTPUT + 'milestone/model_' + COUNTRY + '/'

PATH_TO_TRAINING_DATA = 'data/training_data_complete/'


# preprocessing
WIDTH = 256
HEIGHT = 256
UPPER_CUT = 0.50
LOWER_CUT = 0.50
THREADS = 16
PERCENTILE = 60

MIN_CIRCLE_OFFSET = 10
MAX_CIRCLE_OFFSET = 40


# training
WIDTH_PREPROC = 286
HEIGHT_PREPROC = 286
LAMBDA = 80
OUTPUT_CHANNELS = 1
BUFFERSIZE = 4000
BATCH_SIZE = 16

# vegetation minecraft
BIOMES = ['minecraft:forest', 'minecraft:flower_forest', 'minecraft:birch_forest', 'minecraft:dark_forest',
          'minecraft:parse_jungle']

WATER_BLOCKS = ['flowing_water', 'water']
NON_SOLID_BLOCKS = WATER_BLOCKS + ['air', 'flowing_lava', 'lava', 'grass', 'dead_bush', 'dandelion', 'poppy']
MINECRAFT_CHUNK_SIZE = 16
WOOD_BLOCKS = ['birch_log']
VEGETATION_BLOCKS = WOOD_BLOCKS + ['birch_leaves']

