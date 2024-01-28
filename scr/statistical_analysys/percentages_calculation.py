from scr.statistical_analysys.areas_calculation import (
    calculate_bb_areas,
    calculate_file_areas_sum,
)


def calculate_areas_percentages(
    bb_labeled: list[list[list]], bb_detected: list[list[list]], img_sizes: list
) -> list:
    """
    Calculates areas and then percentages of area detected/labeled

    Parameters:
    :param bb_labeled: All bounding boxes read from .txt files
    :type bb_labeled: list[list[list]]
    :param bb_detected: All bounding boxes from YOLO object detection
    :type bb_detected: list[list[list]]
    :return: A list with percentages
    :rtype: list[float]
    """
    bb_areas_labeled = calculate_bb_areas(bb=bb_labeled, img_sizes=img_sizes)
    bb_areas_detected = calculate_bb_areas(bb=bb_detected, img_sizes=img_sizes)
    percentages = calculate_percentages(
        areas_labeled=bb_areas_labeled, areas_detected=bb_areas_detected
    )
    return percentages


def calculate_areas_percentages_sum(
    bb_labeled: list[list[list]], bb_detected: list[list[list]], img_sizes: list
) -> list:
    """
    Calculates areas and then percentages of area detected/labeled

    Parameters:
    :param bb_labeled: All bounding boxes read from .txt files
    :type bb_labeled: list[list[list]]
    :param bb_detected: All bounding boxes from YOLO object detection
    :type bb_detected: list[list[list]]
    :return: A list with percentages
    :rtype: list[float]
    """
    bb_areas_labeled = calculate_bb_areas(bb=bb_labeled, img_sizes=img_sizes)
    bb_areas_detected = calculate_bb_areas(bb=bb_detected, img_sizes=img_sizes)

    areas_file_sum_labeled = calculate_file_areas_sum(bb_areas=bb_areas_labeled)
    areas_file_sum_detected = calculate_file_areas_sum(bb_areas=bb_areas_detected)
    percentages_sum = calculate_sum_percentages(
        areas_file_sum_labeled=areas_file_sum_labeled,
        areas_file_sum_detected=areas_file_sum_detected,
    )
    return percentages_sum


def calculate_percentages(
    areas_labeled: list[list[float | int]], areas_detected: list[list[float | int]]
) -> list:
    """
    Calculates areas and then percentages of area detected/labeled

    Parameters:
    :param areas_labeled: Areas of img labeled in a list.
    :type areas_labeled: list[list[float | int]]
    :param areas_detected: Areas of img detected in a list.
    :type areas_detected: list[list[float | int]]
    :return: A list with percentages
    :rtype: list
    """
    percentages = []
    for file_labeled, file_detected in zip(areas_labeled, areas_detected):
        for img_labeled, img_detected in zip(file_labeled, file_detected):
            percentages.append(img_detected / img_labeled)
    return percentages


def calculate_sum_percentages(
    areas_file_sum_labeled: list[float], areas_file_sum_detected: list[float]
) -> list:
    """
    Calculates areas and then percentages of area detected/labeled

    Parameters:
    :param areas_file_sum_labeled: Areas of img labeled in a list.
    :type areas_file_sum_labeled: list[list[float | int]]
    :param areas_file_sum_detected: Areas of img detected in a list.
    :type areas_file_sum_detected: list[list[float | int]]
    :return: A list with percentages
    :rtype: list
    """
    percentages = []
    for img_labeled, img_detected in zip(
        areas_file_sum_labeled, areas_file_sum_detected
    ):
        percentages.append(img_detected / img_labeled)
    return percentages
