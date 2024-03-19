import os
import pickle

import argparse
parser = argparse.ArgumentParser()

# parser.add_argument('camera_detail', type=str, help='camera_detail')
# args = parser.parse_args()
CAMERA_DETAIL = "66"
#
# CONFIG_PATH = f'{CAMERA_DETAIL}.pkl'
#
# with open(CONFIG_PATH, 'rb') as file_:
#     config_dict = pickle.load(file_)

CAMERA_NO = 1
# VIDEO_INPUT = 'atm1_20220727054459.avi'
# VIDEO_INPUT = 'rtsp://admin:bmm%4012345@192.168.1.94/Streaming/Channels/201'
VIDEO_INPUT = 'ip-94_19-03-2024_12-43_10_mins_trim.mp4'

# CAMERA_DETAIL = config_dict['camera_detail'] ### unique name

file_fps = 25

SKIP_FRAME = 24

# CROPPING_COORD_PRI = config_dict['crop_coord'] #x1, y1, x2, y2
# [136:1520, 544:1928
#[136:1520, 544:1928]
#CROPPING_COORD_PRI = [0, 360, 140, 500]
# 300:1440, 1300:2440
CROPPING_COORD_PRI = [1300, 300, 2440, 1440]
# CROPPING_COORD_PRI = [144, 136, 1528, 1520]
PADDING = False
#840, 0, 1920, 1080

MIN_CONTOUR_AREA = 1
CONFIDENCE_THRESH = 0.4

MODEL_TYPE = 'TFLITE'

# HEIGHT, WIDTH = config_dict['model_dim']
MODEL_PATH = 'models/pretrained.tflite'

# INPUT_SHAPE = [320, 320]
INPUT_SHAPE = [300, 300]
print(INPUT_SHAPE)

# Counting parameter

# LINE = config_dict['line']
# ENTRY_POINT = config_dict['entry_point']

DRAW = True
VIDEO_WRITE = False


if VIDEO_WRITE:
    VIDEO_WRITE_WIDTH = 300
    VIDEO_WRITE_HEIGHT = 300
    if VIDEO_INPUT == 0:
        VIDEO_INPUT = 'webcam'
    output_filename = '/tmp/'+VIDEO_INPUT.split('/')[-1].split('.')[0]+'_'+MODEL_TYPE+ '_' + str(CONFIDENCE_THRESH*100) + '_result2dev.mp4'

    saving_fps = file_fps/SKIP_FRAME




atm_class_names = [
    "person",
    "helmet",
    "face",
    "head"
]

pretrained_class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic_light",
    "fire_hydrant",
    "stop_sign",
    "parking_meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports_ball",
    "kite",
    "baseball_bat",
    "baseball_glove",
    "skateboard",
    "surfboard",
    "tennis_racket",
    "bottle",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot_dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "dining_table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell_phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy_bear",
    "hair_drier",
    "toothbrush",
]
