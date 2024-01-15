import os

from ultralytics import YOLO

from definitions import (
    ARTICLES_ANALYZE_IMAGES_DIR,
    ARTICLES_ANALYZE_LABELS_DIR,
    PAGES_ANALYZE_IMAGES_DIR,
    PAGES_ANALYZE_LABELS_DIR,
)
from scr.statistical_analysys.percentages_calculation import (
    calculate_areas_percentages,
    calculate_areas_percentages_sum,
)
from scr.statistical_analysys.plot_functions import create_bar_plots
from scr.statistical_analysys.helpers import (
    add_img_names_to_boxes,
    convert_bounding_box_to_yolo_format,
    convert_box_txt_to_float,
    read_bounding_boxes,
    sorted_alphanumeric,
    get_img_sizes,
)
from scr.statistical_analysys.intersection_sorting import sort_boxes_intersection


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

    bounding_boxes_img = sort_boxes_intersection(
        bb_labeled=bounding_boxes_txt, bb_detected=bounding_boxes_img
    )

    percentages_sum = calculate_areas_percentages_sum(
        bounding_boxes_txt=bounding_boxes_txt,
        bounding_boxes_img=bounding_boxes_img,
        img_sizes=img_sizes,
    )
    percentages = calculate_areas_percentages(
        bounding_boxes_txt=bounding_boxes_txt,
        bounding_boxes_img=bounding_boxes_img,
        img_sizes=img_sizes,
    )
    create_bar_plots(percentages_sum=percentages_sum, percentages=percentages)


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

    bounding_boxes_img = sort_boxes_intersection(
        bb_labeled=bounding_boxes_txt, bb_detected=bounding_boxes_img
    )
    percentages_sum = calculate_areas_percentages_sum(
        bounding_boxes_txt=bounding_boxes_txt,
        bounding_boxes_img=bounding_boxes_img,
        img_sizes=img_sizes,
    )
    percentages = calculate_areas_percentages(
        bounding_boxes_txt=bounding_boxes_txt,
        bounding_boxes_img=bounding_boxes_img,
        img_sizes=img_sizes,
    )
    create_bar_plots(percentages_sum=percentages_sum, percentages=percentages)


def get_bounding_boxes_from_txt(txt_input_path: str):
    iterate = 0
    bounding_boxes_txt = []
    for file in sorted_alphanumeric(os.listdir(txt_input_path)):
        bounding_boxes = read_bounding_boxes(path=txt_input_path + file)
        bounding_boxes_txt.append(bounding_boxes)
        iterate += 1
    bounding_boxes_txt = convert_box_txt_to_float(bounding_boxes_txt=bounding_boxes_txt)
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
