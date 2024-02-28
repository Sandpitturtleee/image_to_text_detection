import copy
import os
import re


import cv2


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


def list_min_val(bb_file_detected_body: list[list[float | int]]) -> list[float | int]:
    """
    Deletes from a list a body image that has the smallest item[1] value. And returns deleted item

    Parameters:
    :bb_file_detected_body: List of detected body images
    :type bb_file_detected_body: list[list[float | int]]
    :return: Deleted item with the smallest value
    :rtype: list[float | int]
    """
    min_val = bb_file_detected_body[0]
    for item in bb_file_detected_body:
        if item[1] < min_val[1]:
            min_val = item
    bb_file_detected_body.remove(min_val)
    return min_val


def create_cols(
    bb_file_detected_body: list[list[float | int]],
) -> list[list[float | int]]:
    """
    Creates columns reflecting typical newspaper layout

    Parameters:
    :bb_file_detected_body: List of detected body images
    :type bb_file_detected_body: list[list[float | int]]
    :return: Sorted elements
    :rtype: list[list[float | int]]
    """
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


def sort_body_elements_in_article(
    bb_file_detected_body: list[list[float | int]],
) -> list[float | int]:
    """
    Sorts body text elements simulating a typical newspaper column layout, sorts elements inside created columns

    Parameters:
    :bb_file_detected_body: List of detected body images
    :type bb_file_detected_body: list[list[float | int]]
    :return: Sorted elements
    :rtype: list[float | int]
    """
    cols = create_cols(bb_file_detected_body=bb_file_detected_body)
    sorted_item = []
    for item in cols:
        insertion_sort(item)
        for item_nested in item:
            sorted_item.append(item_nested)
    return sorted_item


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def insertion_sort(arr):
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
        key = arr[i]

        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i - 1
        while j >= 0 and key[2] < arr[j][2]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
