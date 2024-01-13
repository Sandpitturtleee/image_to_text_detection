import os
import re

import cv2
import pandas as pd
import pybboxes as pbx
from matplotlib.pyplot import show
from pandas import DataFrame, cut
from ultralytics import YOLO

from definitions import (ARTICLES_ANALYZE_IMAGES_DIR,
                         ARTICLES_ANALYZE_LABELS_DIR, PAGES_ANALYZE_IMAGES_DIR,
                         PAGES_ANALYZE_LABELS_DIR)
from scr.statistical_analysys.helpers import (
    add_img_names_to_boxes, calculate_areas, calculate_mismatched_length,
    calculate_mismatched_zeros, calculate_percentages,
    calculate_sum_percentages, convert_bounding_box_to_yolo_format,
    create_bar_plot, get_img_names, read_bounding_boxes, sort_nested_list,
    sorted_alphanumeric, sum_areas)
from scr.statistical_analysys.variables import area_bin_edges, area_bin_labels


def analyze_pages():
    img_sizes = get_img_sizes(img_input_path=PAGES_ANALYZE_IMAGES_DIR)

    bounding_boxes_txt = get_bounding_boxes_from_txt(
        txt_input_path=PAGES_ANALYZE_LABELS_DIR,
    )
    bounding_boxes_img = get_bounding_boxes_from_img(
        model_name="newspaper_best.pt",
        img_input_path=PAGES_ANALYZE_IMAGES_DIR,
        img_sizes=img_sizes,
    )

    # analyze_areas(bounding_boxes_txt=bounding_boxes_txt, bounding_boxes_img=bounding_boxes_img, img_sizes=img_sizes)
    analyze_areas_sum(
        bounding_boxes_txt=bounding_boxes_txt,
        bounding_boxes_img=bounding_boxes_img,
        img_sizes=img_sizes,
    )


def analyze_articles():
    img_sizes = get_img_sizes(img_input_path=ARTICLES_ANALYZE_IMAGES_DIR)

    bounding_boxes_txt = get_bounding_boxes_from_txt(
        txt_input_path=ARTICLES_ANALYZE_LABELS_DIR,
    )
    bounding_boxes_img = get_bounding_boxes_from_img(
        model_name="article_best.pt",
        img_input_path=ARTICLES_ANALYZE_IMAGES_DIR,
        img_sizes=img_sizes,
    )
    # analyze_areas(bounding_boxes_txt=bounding_boxes_txt, bounding_boxes_img=bounding_boxes_img, img_sizes=img_sizes)
    analyze_areas_sum(
        bounding_boxes_txt=bounding_boxes_txt,
        bounding_boxes_img=bounding_boxes_img,
        img_sizes=img_sizes,
    )


def get_bounding_boxes_from_txt(txt_input_path: str):
    iterate = 0
    bounding_boxes_txt = []
    for file in sorted_alphanumeric(os.listdir(txt_input_path)):
        bounding_boxes = read_bounding_boxes(path=txt_input_path + file)
        bounding_boxes_txt.append(bounding_boxes)
        iterate += 1
    return bounding_boxes_txt


def get_bounding_boxes_from_img(model_name: str, img_input_path: str, img_sizes: list):
    iterate = 0
    bounding_boxes_img = []
    model = YOLO(model_name)
    for file in sorted_alphanumeric(os.listdir(img_input_path)):
        results = model(img_input_path + file, device="mps")
        box = results[0].boxes.xyxy.tolist()
        bounding_boxes = add_img_names_to_boxes(results=results, bounding_boxes=box)
        bounding_boxes = convert_bounding_box_to_yolo_format(
            voc_bbox=bounding_boxes,
            img_width=img_sizes[iterate][0],
            img_height=img_sizes[iterate][1],
        )
        bounding_boxes_img.append(bounding_boxes)
        iterate += 1
    return bounding_boxes_img


def analyze_areas(bounding_boxes_txt: list, bounding_boxes_img: list, img_sizes: list):
    areas_txt = calculate_areas(bounding_boxes=bounding_boxes_txt, img_sizes=img_sizes)
    areas_img = calculate_areas(bounding_boxes=bounding_boxes_img, img_sizes=img_sizes)

    areas_txt, areas_img, mismatching_length = calculate_mismatched_length(
        areas_txt=areas_txt, areas_img=areas_img
    )
    areas_txt, areas_img, mismatching_zeros = calculate_mismatched_zeros(
        areas_txt=areas_txt, areas_img=areas_img
    )
    areas_txt = sort_nested_list(nested=areas_txt)
    areas_img = sort_nested_list(nested=areas_img)
    percentages = calculate_percentages(areas_txt=areas_txt, areas_img=areas_img)
    create_bar_plot(
        data=percentages, bin_edges=area_bin_edges, bin_labels=area_bin_labels
    )
    # print("Mismatching length = " + str(len(mismatching_length[0])))
    # print("Mismatching zeros = "+str(len(mismatching_zeros[0])))
    # print("Correct areas = " + str(len(areas_txt)))


def analyze_areas_sum(
    bounding_boxes_txt: list, bounding_boxes_img: list, img_sizes: list
):
    areas_txt = calculate_areas(bounding_boxes=bounding_boxes_txt, img_sizes=img_sizes)
    areas_img = calculate_areas(bounding_boxes=bounding_boxes_img, img_sizes=img_sizes)
    areas_txt_sum = sum_areas(areas=areas_txt)
    areas_img_sum = sum_areas(areas=areas_img)
    percentages = calculate_sum_percentages(
        areas_txt=areas_txt_sum, areas_img=areas_img_sum
    )
    create_bar_plot(
        data=percentages, bin_edges=area_bin_edges, bin_labels=area_bin_labels
    )
    # print("Mismatching length = " + str(len(mismatching_length[0])))
    # print("Mismatching zeros = "+str(len(mismatching_zeros[0])))
    # print("Correct areas = " + str(len(areas_txt)))


def get_img_sizes(img_input_path: str):
    img_sizes = []
    for file in sorted_alphanumeric(os.listdir(img_input_path)):
        im = cv2.imread(img_input_path + file)
        h, w, _ = im.shape
        img_sizes.append([w, h])
    return img_sizes
