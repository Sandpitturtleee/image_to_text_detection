import os
import re

import cv2


def read_bounding_boxes(file_name: str) -> list[list[float]]:
    """
    Reads bounding boxes data from txt file and puts it into a list

    Parameters:
    :param file_name: Name of the file
    :type file_name: str
    :return: A list of list with bounding boxes
    :rtype: list[list[float]]
    """
    with open(file_name) as f:
        content = f.read().splitlines()
    content_split = [word for line in content for word in line.split()]
    content_split = to_matrix(content_split, 5)
    return content_split


def convert_bb_file_voc_to_yolo(
    bb_file_voc: list[list[list[float | int]]], img_width: int, img_height: int
) -> list[list[float | int]]:
    """
    Converts bounding box of img images from file from [x1,y1,x2,y2] to [x,y,w,h] format

    Parameters:
    :param bb_file_voc: List with bounding boxes for all images in a file
    :type bb_file_voc: list[list[float]]
    :param img_width: Height of a base file image
    :type img_width: int
    :param img_height: Height of a base file image
    :type img_height: int
    :return: Bounding boxes in yolo format
    :rtype: list[list[float|int]]
    """
    bb_file_voc_converted = []
    for bb_img_voc in bb_file_voc:
        bb_file_voc_converted.append(
            list(
                convert_bb_img_voc_to_yolo(
                    bb_img_voc=bb_img_voc, img_width=img_width, img_height=img_height
                )
            )
        )

    return bb_file_voc_converted


def convert_bb_img_voc_to_yolo(
    bb_img_voc: list[list[float | int]], img_width: int, img_height: int
) -> list[float | int]:
    """
    Converts bounding box of a single img from [x1,y1,x2,y2] to [x,y,w,h] format

    Parameters:
    :param bb_img_voc: List with bounding box for one img file
    :type bb_img_voc: list[list[float|int]]
    :param img_width: Height of a base file image
    :type img_width: int
    :param img_height: Height of a base file image
    :type img_height: int
    :return: Bounding boxes in yolo format
    :rtype: list[float|int]
    """
    bb_img_converted = convert_bb_img_detected_edges(bb_img_detected=bb_img_voc)
    name, x1, y1, x2, y2 = bb_img_converted
    return [
        name,
        ((x2 + x1) / (2 * img_width)),
        ((y2 + y1) / (2 * img_height)),
        (x2 - x1) / img_width,
        (y2 - y1) / img_height,
    ]


def convert_bb_img_detected_edges(bb_img_detected: list[list[float | int]]) -> list:
    """
    Converts x1 x2 y1 y2 to correct box format, changing edges when YOLO gave wrong result

    Parameters:
    :param bb_img_detected: List with bounding box for one img file
    :type bb_img_detected: list[list[float|int]]
    :return: Converted bounding box
    :rtype: list[float|int]
    """
    name, x1, y1, x2, y2 = bb_img_detected
    x1_converted = min(x1, x2)
    y1_converted = min(y1, y2)
    x2_converted = max(x1, x2)
    y2_converted = max(y1, y2)
    return [name, x1_converted, y1_converted, x2_converted, y2_converted]


def convert_bb_labeled_to_float(
    bb_labeled: list[list[list[float]]],
) -> list[list[list[float]]]:
    """
    Converts x1 x2 y1 y2 to correct box format, changing edges when YOLO gave wrong result

    Parameters:
    :param bb_labeled: List with bounding boxes in string
    :type bb_labeled: list[list[list[]]]
    :return: Bounding boxes converted to float
    :rtype: list[list[list[]]]
    """
    bb_labeled_converted = []
    for bb_file_labeled in bb_labeled:
        bb_file_labeled_converted = []
        for bb_img_labeled in bb_file_labeled:
            bb_file_labeled_converted.append([float(i) for i in bb_img_labeled])
        bb_labeled_converted.append(bb_file_labeled_converted)
    return bb_labeled_converted


def get_img_sizes(folder_path: str) -> list[list[int]]:
    """
    Gets sizes of all images from folder

    Parameters:
    :param folder_path: Path to folder with images
    :type folder_path: str
    :return: A list with sizes
    :rtype: list[list[int]]
    """
    img_sizes = []
    for file in sorted_alphanumeric(os.listdir(folder_path)):
        im = cv2.imread(folder_path + file)
        h, w, _ = im.shape
        img_sizes.append([w, h])
    return img_sizes


def get_img_detected_classes(results: list) -> list:
    """
    Gets classes numbers 0,1,2... detected for file image

    Parameters:
    :param results: Expected for YOLO object detection on file
    :type results: list
    :return: A list with detected classes
    :rtype: list
    """
    img_detected_classes = []
    for r in results:
        for c in r.boxes.cls:
            img_detected_classes.append(int(c))
    return img_detected_classes


def add_img_names_to_boxes(
    results: list, bb_labeled: list[list[list[float]]]
) -> list[list[list[float]]]:
    """
    Ads classes numbers 0,1,2... detected for file image to bounding boxes

    Parameters:
    :param results: Expected for YOLO object detection on file
    :type results: list
    :param bb_labeled: Expected for YOLO object detection on file
    :type bb_labeled: list[list[list[float]]]
    :return: A list with detected classes
    :rtype: list
    """
    img_detected_classes = get_img_detected_classes(results=results)
    for i in range(len(bb_labeled)):
        bb_labeled[i].insert(0, img_detected_classes[i])
    return bb_labeled


def intersection_yolo(
    bb_img_base: list[float | int], bb_img: list[float | int]
) -> list[float | int] | int:
    """
    Intersection box of two bounding boxes in YOLO [x,y,w,h] format

    Parameters:
    :param bb_img_base: Bounding box of base image (full size)
    :type bb_img_base: list[float|int]
    :param bb_img: Bounding box of second image
    :type bb_img: list[float|int]
    :return: Intersected bounding box
    :rtype: list[float|int]|int
    """
    try:
        name, x, y, w, h = (
            bb_img[0],
            bb_img[1],
            bb_img[2],
            bb_img[3],
            bb_img[4],
        )
        a_x, a_y, a_w, a_h = (
            float(bb_img_base[0]),
            float(bb_img_base[1]),
            float(bb_img_base[2]),
            float(bb_img_base[3]),
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


def to_matrix(list_to_convert: list, split_length: int):
    return [
        list_to_convert[i : i + split_length]
        for i in range(0, len(list_to_convert), split_length)
    ]


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)
