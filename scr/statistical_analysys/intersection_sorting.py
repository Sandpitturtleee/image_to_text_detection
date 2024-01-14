from numpy import array, ndarray, unravel_index


def sort_boxes_intersection(
    bb_labeled: list[list[list]], bb_detected: list[list[list]]
) -> list[list[list]]:
    """
    Sorts bb_detected according to correct labels in bb_labeled based on intersection over union
    of each individual element in both lists.
    If bb_file_labeled > bb_file_detected ex. 11 > 7 second list is populated with zeros in remaining places
    for i in range(len(bb_file_labeled)):
        if i == len(bb_file_detected):
            break
    Is used in an example above to stop a loop from adding too much zeros to bb_sorted_detected

    Parameters:
    :param bb_labeled: All bounding boxes read from .txt files
    :type bb_labeled: list[list[list]]
    :param bb_detected: All bounding boxes from YOLO object detection
    :type bb_detected: list[list[list]]
    :return: A list of list containing intersection percentages
    :rtype: list[list[list]]
    """
    bb_sorted_detected = []
    for bb_file_labeled, bb_file_detected in zip(bb_labeled, bb_detected):
        bb_file_sorted_detected = create_blank_list_of_length(
            bb_file_labeled=bb_file_labeled
        )
        intersection_file_detected = array(
            create_intersection_file_detected(
                bb_file_labeled=bb_file_labeled, bb_file_detected=bb_file_detected
            )
        )
        for i in range(len(bb_file_labeled)):
            if i == len(bb_file_detected):
                break
            max_index_intersection_file_detected = unravel_index(
                intersection_file_detected.argmax(), intersection_file_detected.shape
            )
            bb_file_sorted_detected[
                max_index_intersection_file_detected[0]
            ] = bb_file_detected[max_index_intersection_file_detected[1]]
            bb_file_detected[max_index_intersection_file_detected[1]] = 0
            intersection_file_detected = populate_intersection_file_detected_with_zeros(
                max_index_intersection_file_detected=max_index_intersection_file_detected,
                intersection_file_detected=intersection_file_detected,
            )

        add_bb_img_detected(
            bb_file_detected=bb_file_detected,
            bb_file_sorted_detected=bb_file_sorted_detected,
        )
        bb_sorted_detected.append(bb_file_sorted_detected)
    return bb_sorted_detected


def create_intersection_file_detected(
    bb_file_labeled: list[list], bb_file_detected: list[list]
) -> list[list]:
    """
    Creates a 2d list of intersection percentages of each labeled img in bb_file_labeled over each detected img
    in bb_file_detected. Size of a list is len(bb_file_labeled) X len(bb_detected_labeled)

    Parameters:
    :param bb_file_labeled: A list of list containing bounding boxes of images labeled in a file
    :type bb_file_labeled: list[list]
    :param bb_file_detected: A list of list containing bounding boxes of images detected in a file
    :type bb_file_detected: list[list]
    :return: A list of list containing intersection percentages
    :rtype: list[list]
    """
    intersection_file_detected = []
    for bb_img_labeled in bb_file_labeled:
        intersection_img_detected = []
        for bb_img_detected in bb_file_detected:
            intersection_img_detected.append(
                bb_intersection_over_union(
                    bb_img_labeled=bb_img_labeled, bb_img_detected=bb_img_detected
                )
            )
        intersection_file_detected.append(intersection_img_detected)
    return intersection_file_detected


def create_blank_list_of_length(bb_file_labeled: list[list]) -> list:
    """
    Creates a list of len(bb_file_labeled) populated with zeros

    Parameters:
    :param bb_file_labeled: A list of list containing bounding boxes of images labeled in a file
    :type bb_file_labeled: list[list]
    :return: A list populated with zeros
    :rtype: list
    """
    new_list = []
    for i in range(len(bb_file_labeled)):
        new_list.append(0)
    return new_list


def populate_intersection_file_detected_with_zeros(
    max_index_intersection_file_detected: tuple, intersection_file_detected: ndarray
) -> ndarray:
    """
    Populated intersection_file_detected with zeros in a correct rows and columns according to
    max_index_intersection_file_detected.
    For example max_index_intersection_file_detected = (4,5) - row index 4, column index 5

    Parameters:
    :param max_index_intersection_file_detected: Index of the biggest element in intersection_file_detected
    :type max_index_intersection_file_detected: tuple
    :param intersection_file_detected: A ndarray containing intersection percentages
    :type intersection_file_detected: ndarray
    :return: An ndarray populated with zeros in a correct row in column
    :rtype: ndarray
    """
    for i in range(
        len(intersection_file_detected[max_index_intersection_file_detected[0]])
    ):
        intersection_file_detected[max_index_intersection_file_detected[0]][i] = 0
    for item in intersection_file_detected:
        item[max_index_intersection_file_detected[1]] = 0
    return intersection_file_detected


def bb_intersection_over_union(bb_img_labeled: list, bb_img_detected: list) -> float:
    """
    Calculate intersection over union of two areas defined by YOLO coordinates [name,x,y,w,h]

    Parameters:
    :param bb_img_labeled: Bounding box of labeled img in YOLO format
    :type bb_img_labeled: list
    :param bb_img_detected: Bounding box of detected img in YOLO format
    :type bb_img_detected: list
    :return: intersection over union ex. 0.994564
    :rtype: float
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(bb_img_labeled[1], bb_img_detected[1])
    y_a = max(bb_img_labeled[2], bb_img_detected[2])
    x_b = min(bb_img_labeled[3], bb_img_detected[3])
    y_b = min(bb_img_labeled[4], bb_img_detected[4])
    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (bb_img_labeled[3] - bb_img_labeled[1] + 1) * (
        bb_img_labeled[4] - bb_img_labeled[2] + 1
    )
    box_b_area = (bb_img_detected[3] - bb_img_detected[1] + 1) * (
        bb_img_detected[4] - bb_img_detected[2] + 1
    )
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    # return the intersection over union value
    return iou


def add_bb_img_detected(
    bb_file_detected: list[list], bb_file_sorted_detected: list[list]
):
    """
    Ads missing bb_img_detected to bb_file_sorted_detected it happens when
    len(bb_file_labeled) < len(bb_file_detected)

    Parameters:
    :param bb_file_detected: A list of list containing bounding boxes of images detected in a file
    :type bb_file_detected: list[list]
    :param bb_file_sorted_detected: A list of list containing bounding boxes of images detected in a file, sorted
    :type bb_file_sorted_detected: list[list]
    """
    for bb_img_detected in bb_file_detected:
        if bb_img_detected not in bb_file_sorted_detected and bb_img_detected != 0:
            bb_file_sorted_detected.append(bb_img_detected)
