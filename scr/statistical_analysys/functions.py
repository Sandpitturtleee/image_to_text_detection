import os
import re

from definitions import PAGES_TRAIN_LABELS_DIR
from natsort import natsorted


def load_files_and_calculate_areas(path: str):
    areas = []
    for file in sorted_alphanumeric(os.listdir(path)):
        bounding_boxes = read_bounding_boxes(
            path=path + file)
        areas.append(calculate_area(img_width=640, img_length=640, bounding_boxes=bounding_boxes))
    return areas


def read_bounding_boxes(path: str):
    with open(path) as f:
        content = f.read().splitlines()
    content_split = [word for line in content for word in line.split()]
    content_split = to_matrix(content_split, 5)
    return content_split


def calculate_area(img_width: int, img_length: int, bounding_boxes: list):
    areas = []
    for item in bounding_boxes:
        areas.append(float(item[3]) * img_width * float(item[4]) * img_length)
    return areas


def to_matrix(list_to_convert: list, split_length: int):
    return [list_to_convert[i: i + split_length] for i in range(0, len(list_to_convert), split_length)]


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)
