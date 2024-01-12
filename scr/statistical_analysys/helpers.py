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


def calculate_areas(bounding_boxes: list, img_sizes: list, i: int):
    areas = []
    iterate = 0
    for item in bounding_boxes:
        areas.append(
            calculate_area(
                img_width=img_sizes[iterate][0],
                img_height=img_sizes[iterate][1],
                bounding_boxes=item,
                i=i,
            )
        )
        iterate += 1
    return areas


def calculate_area(img_width: int, img_height: int, bounding_boxes: list, i: int):
    areas = []
    for item in bounding_boxes:
        areas.append(float(item[i]) * img_width * float(item[i + 1]) * img_height)
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
    try:
        for item in voc_bbox:
            converted_voc_bbox.append(
                list(
                    pbx.convert_bbox(
                        item,
                        from_type="voc",
                        to_type="yolo",
                        image_size=(img_width, img_height),
                    )
                )
            )
    except ValueError:
        converted_voc_bbox.append([0, 0, 0, 0])

    return converted_voc_bbox


def calculate_percentages(areas_txt: list, areas_img: list):
    percentages = []
    for item_txt, item_img in zip(areas_txt, areas_img):
        for nested_item_txt, nested_item_img in zip(item_txt, item_img):
            percentages.append(nested_item_img / nested_item_txt)
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
