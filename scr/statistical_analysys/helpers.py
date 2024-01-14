import re

import pandas as pd
import pybboxes as pbx
from matplotlib.pyplot import show
from pandas import cut


def sum_areas(areas: list):
    areas_sum = []
    for item in areas:
        areas_sum.append(sum(item))
    return areas_sum


def calculate_areas(bounding_boxes: list, img_sizes: list):
    areas = []
    iterate = 0
    for item in bounding_boxes:
        areas.append(
            calculate_area(
                img_width=img_sizes[iterate][0],
                img_height=img_sizes[iterate][1],
                bounding_boxes=item,
            )
        )
        iterate += 1
    return areas


def calculate_area(img_width: int, img_height: int, bounding_boxes: list):
    areas = []
    base_box = [0.5, 0.5, 1, 1]
    for item in bounding_boxes:
        new_item = intersection_yolo(base_box=base_box, box=item)
        try:
            areas.append(float(new_item[3]) * img_width * float(new_item[4]) * img_height)
        except TypeError:
            # Happens when txt_size>img_size 11>7
            areas.append(0)
    return areas


def calculate_mismatched_length(areas_txt: list, areas_img: list):
    mismatching_length = 0
    mismatching_length_img = []
    mismatching_length_txt = []
    for item_txt, item_img in zip(areas_txt, areas_img):
        if len(item_txt) != len(item_img):
            mismatching_length += 1
            mismatching_length_txt.append(item_txt)
            mismatching_length_img.append(item_img)
            areas_txt.remove(item_txt)
            areas_img.remove(item_img)
    return areas_txt, areas_img, [mismatching_length_txt, mismatching_length_img]


def calculate_mismatched_zeros(areas_txt: list, areas_img: list):
    mismatching_zeros = 0
    mismatching_zeros_img = []
    mismatching_zeros_txt = []
    for item_txt, item_img in zip(areas_txt, areas_img):
        if 0 in item_img:
            mismatching_zeros += 1
            mismatching_zeros_txt.append(item_txt)
            mismatching_zeros_img.append(item_img)
            areas_txt.remove(item_txt)
            areas_img.remove(item_img)
    return areas_txt, areas_img, [mismatching_zeros_txt, mismatching_zeros_img]


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
    name, x1, y1, x2, y2 = box
    return [
        name,
        abs(((x2 + x1) / (2 * image_w))),
        abs(((y2 + y1) / (2 * image_h))),
        abs((x2 - x1) / image_w),
        abs((y2 - y1) / image_h),
    ]


def calculate_percentages(areas_txt: list, areas_img: list):
    percentages = []
    for item_txt, item_img in zip(areas_txt, areas_img):
        for nested_item_txt, nested_item_img in zip(item_txt, item_img):
            result = nested_item_img/nested_item_txt
            if result != 0:
                percentages.append(result)
    return percentages


def calculate_sum_percentages(areas_txt: list, areas_img: list):
    percentages = []
    for item_txt, item_img in zip(areas_txt, areas_img):
        percentages.append(item_img / item_txt)
    return percentages


def read_bounding_boxes(path: str):
    with open(path) as f:
        content = f.read().splitlines()
    content_split = [word for line in content for word in line.split()]
    content_split = to_matrix(content_split, 5)
    return content_split


def create_bar_plot(data: list, bin_edges: list, bin_labels: list):
    columns = f"{data=}".split("=")[0]
    df = pd.DataFrame(data, columns=[columns])
    df["bin"] = cut(
        df[columns],
        [
            *bin_edges,
            float("inf"),
        ],
        labels=bin_labels,
    )
    df["bin"].value_counts().sort_index().plot.bar()
    show()


def to_matrix(list_to_convert: list, split_length: int):
    return [
        list_to_convert[i : i + split_length]
        for i in range(0, len(list_to_convert), split_length)
    ]


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def sort_nested_list(nested: list):
    for item in nested:
        item.sort()
    return nested


def intersection_yolo(base_box: list, box: list):
    try:
        name, x, y, w, h = (
            box[0],
            box[1],
            box[2],
            box[3],
            box[4],
        )
        a_x, a_y, a_w, a_h = (
            float(base_box[0]),
            float(base_box[1]),
            float(base_box[2]),
            float(base_box[3]),
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


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[3] - boxA[1] + 1) * (boxA[4] - boxA[2] + 1)
    boxBArea = (boxB[3] - boxB[1] + 1) * (boxB[4] - boxB[2] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def convert_box_txt_to_float(bounding_boxes_txt):
    bounding_boxes_txt_converted = []
    for box_txt in bounding_boxes_txt:
        box_txt_converted = []
        for item_txt in box_txt:
            box_txt_converted.append([float(i) for i in item_txt])
        bounding_boxes_txt_converted.append(box_txt_converted)
    return bounding_boxes_txt_converted


def create_2d_intersection_percentage_list(box_txt: list,box_img: list):
    intersection_percentage_2d = []
    for item_txt in box_txt:
        intersection_percentage = []
        for item_img in box_img:
            intersection_percentage.append(
                bb_intersection_over_union(item_txt, item_img)
            )
        intersection_percentage_2d.append(intersection_percentage)
    return intersection_percentage_2d

def create_blank_list_of_length(box_txt: list):
    blank_list = []
    for i in range(len(box_txt)):
        blank_list.append(0)
    return blank_list

def insert_0_to_list(max_index_2d: tuple, list_2d: list):
    for i in range(len(list_2d[max_index_2d[0]])):
        list_2d[max_index_2d[0]][i] = 0
    for item in list_2d:
        item[max_index_2d[1]] = 0
    return list_2d
