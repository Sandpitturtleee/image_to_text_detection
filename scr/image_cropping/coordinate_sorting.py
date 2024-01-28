import copy
import os
import re

import cv2


def coordinate_file_sorting(bb_file_detected_body, img_width, img_height):
    bb_file_detected_body = convert_bb_file_voc_to_yolo(
        bb_file_voc=bb_file_detected_body, img_width=img_width, img_height=img_height
    )
    return bb_file_detected_body


def convert_bb_file_voc_to_yolo(
    bb_file_voc: list[list[list[float | int]]], img_width, img_height
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
    # img_sizes = get_img_sizes(
    #     "/Users/sold/Desktop/Python/Projects/image_to_text_detection/database/detect/4_articles_cropped/")
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


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def list_min_val(bb_file_detected_body):
    min_val = bb_file_detected_body[0]
    for item in bb_file_detected_body:
        if item[1] < min_val[1]:
            min_val = item
    bb_file_detected_body.remove(min_val)
    return min_val

def create_cols(bb_file_detected_body):
    xy_new = copy.deepcopy(bb_file_detected_body)
    xy_new1 = copy.deepcopy(bb_file_detected_body)
    cols = [[]]
    min_val = list_min_val(bb_file_detected_body=xy_new1)
    min_val_old = min_val
    col_iter = 0
    for _ in bb_file_detected_body:
        min_val = list_min_val(bb_file_detected_body=xy_new)
        diff = abs(min_val[1] - min_val_old[1])
        if diff < 0.10:
            cols[col_iter].append(min_val)
        else:
            cols.append([])
            col_iter += 1
            cols[col_iter].append(min_val)
        min_val_old = min_val
    return cols


def bubbleSort(array):
    # loop through each element of array
    for i in range(len(array)):

        # keep track of swapping
        swapped = False

        # loop to compare array elements
        for j in range(0, len(array) - i - 1):

            # compare two adjacent elements
            # change > to < to sort in descending order
            if array[j][2] > array[j + 1][2]:
                # swapping occurs if elements
                # are not in the intended order
                temp = array[j][2]
                array[j][2] = array[j + 1][2]
                array[j + 1][2] = temp

                swapped = True

        # no swapping means the array is already sorted
        # so no need for further comparison
        if not swapped:
            break

def sort_body_elements_in_article(bb_file_detected_body):
    cols = create_cols(bb_file_detected_body=bb_file_detected_body)
    # pprint(cols)
    for item in cols:
        sorted_item = []
        bubbleSort(item)
    return cols