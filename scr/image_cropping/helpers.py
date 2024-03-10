
def add_img_names_to_boxes(
    names: list, results: list, bb_labeled: list[list[list[float]]]
) -> list[list[list[float]]]:
    """
    Ads classes numbers 0,1,2... detected for file image to bounding boxes

    Parameters:
    :param names: Names of files
    :type names: list
    :param results: Expected for YOLO object detection on file
    :type results: list
    :param bb_labeled: Expected for YOLO object detection on file
    :type bb_labeled: list[list[list[float]]]
    :return: A list with detected classes
    :rtype: list
    """
    img_detected_classes = get_img_detected_classes(names=names, results=results)
    for i in range(len(bb_labeled)):
        bb_labeled[i].insert(0, img_detected_classes[i])
    return bb_labeled


def get_img_detected_classes(names: list, results: list) -> list:
    """
    Gets classes numbers 0,1,2... detected for file image

    Parameters:
    :param names: Names of files
    :type names: list
    :param results: Expected for YOLO object detection on file
    :type results: list
    :return: A list with detected classes
    :rtype: list
    """
    detected_classes = []
    for r in results:
        for c in r.boxes.cls:
            detected_classes.append(names[int(c)])

    return detected_classes


def divide_bb_file_detected(
    bb_file_detected: list[list[list[float | int]]],
) -> tuple[list[list[list[float | int]]], list[list[list[float | int]]]]:
    """
    Creates a two list of detected images divided by "body" and other types

    Parameters:
    :param bb_file_detected: A list o detected images
    :type bb_file_detected: list[list[list[float | int]]]
    :return: A list with detected classes
    :rtype: tuple[list[list[list[float | int]]], list[list[list[float | int]]]]
    """
    bb_img_detected_body = []
    bb_img_detected_other = []
    for bb_img_detected in bb_file_detected:
        if bb_img_detected[0] == "body":
            bb_img_detected_body.append(bb_img_detected)
        else:
            bb_img_detected_other.append(bb_img_detected)
    return bb_img_detected_body, bb_img_detected_other
