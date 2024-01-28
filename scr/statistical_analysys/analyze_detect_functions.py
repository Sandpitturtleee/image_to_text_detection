import os

from ultralytics import YOLO

from definitions import (
    ARTICLES_ANALYZE_IMAGES_DIR,
    ARTICLES_ANALYZE_LABELS_DIR,
    PAGES_ANALYZE_IMAGES_DIR,
    PAGES_ANALYZE_LABELS_DIR,
)
from scr.statistical_analysys.helpers import (
    add_img_names_to_boxes,
    convert_bb_file_voc_to_yolo,
    convert_bb_labeled_to_float,
    get_img_sizes,
    read_bounding_boxes,
    sorted_alphanumeric,
)
from scr.statistical_analysys.intersection_sorting import sort_boxes_intersection
from scr.statistical_analysys.percentages_calculation import (
    calculate_areas_percentages,
    calculate_areas_percentages_sum,
)
from scr.statistical_analysys.plot_functions import create_bar_plots


def analyze_pages():
    """
    Big function grouping al logic from module, gets bb from files then do operations to create plots
    For pages
    """
    img_sizes = get_img_sizes(folder_path=PAGES_ANALYZE_IMAGES_DIR)

    bb_labeled = get_bounding_boxes_from_txt(
        folder_path=PAGES_ANALYZE_LABELS_DIR,
    )
    bb_detected = get_bounding_boxes_from_img(
        model_name="newspaper_best.pt",
        folder_path=PAGES_ANALYZE_IMAGES_DIR,
        img_sizes=img_sizes,
    )

    bb_detected = sort_boxes_intersection(
        bb_labeled=bb_labeled, bb_detected=bb_detected
    )

    percentages_sum = calculate_areas_percentages_sum(
        bb_labeled=bb_labeled,
        bb_detected=bb_detected,
        img_sizes=img_sizes,
    )
    percentages = calculate_areas_percentages(
        bb_labeled=bb_labeled,
        bb_detected=bb_detected,
        img_sizes=img_sizes,
    )
    create_bar_plots(percentages_sum=percentages_sum, percentages=percentages)


def analyze_articles():
    """
    Big function grouping al logic from module, gets bb from files then do operations to create plots
    For articles
    """
    img_sizes = get_img_sizes(folder_path=ARTICLES_ANALYZE_IMAGES_DIR)

    bb_labeled = get_bounding_boxes_from_txt(
        folder_path=ARTICLES_ANALYZE_LABELS_DIR,
    )
    bb_detected = get_bounding_boxes_from_img(
        model_name="article_best.pt",
        folder_path=ARTICLES_ANALYZE_IMAGES_DIR,
        img_sizes=img_sizes,
    )

    bb_detected = sort_boxes_intersection(
        bb_labeled=bb_labeled, bb_detected=bb_detected
    )
    percentages_sum = calculate_areas_percentages_sum(
        bb_labeled=bb_labeled,
        bb_detected=bb_detected,
        img_sizes=img_sizes,
    )
    percentages = calculate_areas_percentages(
        bb_labeled=bb_labeled,
        bb_detected=bb_detected,
        img_sizes=img_sizes,
    )
    create_bar_plots(percentages_sum=percentages_sum, percentages=percentages)


def get_bounding_boxes_from_txt(folder_path: str) -> list[list[list[float | int]]]:
    """
    Gets bounding boxes from txt labeled files

    Parameters:
    :param folder_path: Path to folder
    :type folder_path: str
    :return: A list all labeled bounding boxes
    :rtype: list[list[list[float|int]]]
    """
    iterate = 0
    bb_labeled = []
    for file in sorted_alphanumeric(os.listdir(folder_path)):
        bb_file_labeled = read_bounding_boxes(file_name=folder_path + file)
        bb_labeled.append(bb_file_labeled)
        iterate += 1
    bb_labeled = convert_bb_labeled_to_float(bb_labeled=bb_labeled)
    return bb_labeled


def get_bounding_boxes_from_img(
    model_name: str, folder_path: str, img_sizes: list
) -> list[list[list[float | int]]]:
    """
    Gets bounding boxes from img detected files

    Parameters:
    :param model_name: Model file name
    :type model_name: str
    :param folder_path: Path to folder with images
    :type folder_path: str
    :param img_sizes: List with sizes of images
    :type img_sizes: str
    :return: A list all detected bounding boxes
    :rtype: list[list[list[float|int]]]
    """
    iterate = 0
    bb_detected = []
    model = YOLO(model_name)
    for file in sorted_alphanumeric(os.listdir(folder_path)):
        results = model(folder_path + file, device="mps")
        box = results[0].boxes.xyxy.tolist()
        bb_file_detected = add_img_names_to_boxes(results=results, bb_labeled=box)
        bb_file_detected = convert_bb_file_voc_to_yolo(
            bb_file_voc=bb_file_detected,
            img_width=img_sizes[iterate][0],
            img_height=img_sizes[iterate][1],
        )
        bb_detected.append(bb_file_detected)
        iterate += 1
    return bb_detected
