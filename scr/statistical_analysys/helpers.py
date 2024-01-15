import os
import re

import cv2


def convert_bounding_box_to_yolo_format(
    voc_bbox: list, img_width: int, img_height: int
):
    converted_voc_bbox = []

    for item in voc_bbox:
        converted_voc_bbox.append(
            list(pascal_voc_to_yolo(box=item, image_w=img_width, image_h=img_height))
        )

    return converted_voc_bbox


def pascal_voc_to_yolo(box, image_w, image_h):
    converted_box = convert_bb_img_detected_edges(bb_img_detected=box)
    name, x1, y1, x2, y2 = converted_box
    return [
        name,
        ((x2 + x1) / (2 * image_w)),
        ((y2 + y1) / (2 * image_h)),
        (x2 - x1) / image_w,
        (y2 - y1) / image_h,
    ]


def convert_bb_img_detected_edges(bb_img_detected: list) -> list:
    name, x1, y1, x2, y2 = bb_img_detected
    x1_converted = min(x1, x2)
    y1_converted = min(y1, y2)
    x2_converted = max(x1, x2)
    y2_converted = max(y1, y2)
    return [name, x1_converted, y1_converted, x2_converted, y2_converted]


def read_bounding_boxes(path: str):
    with open(path) as f:
        content = f.read().splitlines()
    content_split = [word for line in content for word in line.split()]
    content_split = to_matrix(content_split, 5)
    return content_split


def to_matrix(list_to_convert: list, split_length: int):
    return [
        list_to_convert[i : i + split_length]
        for i in range(0, len(list_to_convert), split_length)
    ]


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def intersection_yolo(bb_base: list, bb_img: list):
    try:
        name, x, y, w, h = (
            bb_img[0],
            bb_img[1],
            bb_img[2],
            bb_img[3],
            bb_img[4],
        )
        a_x, a_y, a_w, a_h = (
            float(bb_base[0]),
            float(bb_base[1]),
            float(bb_base[2]),
            float(bb_base[3]),
        )
        a_max_x = a_x + a_w / 2
        a_min_x = a_x - a_w / 2

        b_max_x = x + w / 2
        b_min_x = x - w / 2

        c_min_x = max(a_min_x, b_min_x)
        c_max_x = min(a_max_x, b_max_x)
        c_len_x = c_max_x - c_min_x

        a_max_y = a_y + a_h / 2
        a_min_y = a_y - a_h / 2

        b_max_y = y + h / 2
        b_min_y = y - h / 2

        c_min_y = max(a_min_y, b_min_y)
        c_max_y = min(a_max_y, b_max_y)
        c_len_y = c_max_y - c_min_y
        area = c_len_y * c_len_x

        c_w = c_len_x
        c_h = c_len_y
        c_x = c_min_x + 0.5 * c_w
        c_y = c_min_y + 0.5 * c_h
    except TypeError:
        # Happens when txt_size>img_size 11>7
        return 0
    return [name, c_x, c_y, c_w, c_h]


def get_img_names(results: list):
    img_names = []
    for r in results:
        for c in r.boxes.cls:
            img_names.append(int(c))
    return img_names


def add_img_names_to_boxes(results: list, bounding_boxes: list):
    img_names = get_img_names(results=results)
    for i in range(len(bounding_boxes)):
        bounding_boxes[i].insert(0, img_names[i])
    return bounding_boxes


def convert_box_txt_to_float(bounding_boxes_txt):
    bounding_boxes_txt_converted = []
    for box_txt in bounding_boxes_txt:
        box_txt_converted = []
        for item_txt in box_txt:
            box_txt_converted.append([float(i) for i in item_txt])
        bounding_boxes_txt_converted.append(box_txt_converted)
    return bounding_boxes_txt_converted


def get_img_sizes(img_input_path: str):
    img_sizes = []
    for file in sorted_alphanumeric(os.listdir(img_input_path)):
        im = cv2.imread(img_input_path + file)
        h, w, _ = im.shape
        img_sizes.append([w, h])
    return img_sizes
