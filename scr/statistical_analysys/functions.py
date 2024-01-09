import os
import re
from ultralytics import YOLO
import cv2


def detect_images(model_name: str, input_path: str):
    model = YOLO(model_name)
    for file in sorted_alphanumeric(os.listdir(input_path)):
        results = model(input_path + file, device="mps")
        # print(file)
        # print(results[0].boxes.xyxy.tolist())


def load_files_and_calculate_areas(txt_input_path: str, img_input_path):
    areas = []
    iter = 0
    img_sizes = get_img_sizes(img_input_path=img_input_path)
    for file in sorted_alphanumeric(os.listdir(txt_input_path)):
        bounding_boxes = read_bounding_boxes(
            path=txt_input_path + file)
        # print(file)
        # print(bounding_boxes)
        # print(calculate_area(img_width=640, img_length=640, bounding_boxes=bounding_boxes))
        # print()
        areas.append(calculate_area(img_width=img_sizes[iter][0], img_length=img_sizes[iter][1], bounding_boxes=bounding_boxes))
        iter += 1

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


def get_img_sizes(img_input_path: str):
    img_sizes = []
    for file in sorted_alphanumeric(os.listdir(img_input_path)):
        im = cv2.imread(img_input_path + file)
        h, w, _ = im.shape
        img_sizes.append([w, h])
    return img_sizes

def to_matrix(list_to_convert: list, split_length: int):
    return [list_to_convert[i: i + split_length] for i in range(0, len(list_to_convert), split_length)]


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)
