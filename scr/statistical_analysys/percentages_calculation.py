from scr.statistical_analysys.areas_calculation import (
    calculate_bb_areas,
    calculate_file_areas_sum,
)


def calculate_areas_percentages(
    bounding_boxes_txt: list, bounding_boxes_img: list, img_sizes: list
):
    areas_txt = calculate_bb_areas(bb=bounding_boxes_txt, img_sizes=img_sizes)
    areas_img = calculate_bb_areas(bb=bounding_boxes_img, img_sizes=img_sizes)
    percentages = calculate_percentages(areas_txt=areas_txt, areas_img=areas_img)
    return percentages


def calculate_areas_percentages_sum(
    bounding_boxes_txt: list, bounding_boxes_img: list, img_sizes: list
):
    areas_txt = calculate_bb_areas(bb=bounding_boxes_txt, img_sizes=img_sizes)
    areas_img = calculate_bb_areas(bb=bounding_boxes_img, img_sizes=img_sizes)

    areas_txt_sum = calculate_file_areas_sum(bb_areas=areas_txt)
    areas_img_sum = calculate_file_areas_sum(bb_areas=areas_img)
    percentages_sum = calculate_sum_percentages(
        areas_txt=areas_txt_sum, areas_img=areas_img_sum
    )
    return percentages_sum


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
