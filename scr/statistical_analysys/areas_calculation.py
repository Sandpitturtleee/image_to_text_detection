from scr.statistical_analysys.helpers import intersection_yolo


def calculate_bb_areas(
    bb: list[list[list]], img_sizes: list[list]
) -> list[list[float | int]]:
    """
    Calculates areas of ar images detected or labeled by bounding boxes in a single file

    Parameters:
    :param bb: Contains bounding boxes for all images for all files in a single list.
    :type bb: list[list[list]]
    :param img_sizes: A list of list with size of file(img), [width,height]
    :type img_sizes: list[list]
    :return: A list with areas of all images in bb
    :rtype: list[list[float|int]]
    """
    bb_areas = []
    for i in range(len(bb)):
        bb_areas.append(
            calculate_bb_file_areas(
                img_width=img_sizes[i][0],
                img_height=img_sizes[i][1],
                bb_file=bb[i],
            )
        )
    return bb_areas


def calculate_bb_file_areas(
    img_width: int, img_height: int, bb_file: list
) -> list[float | int]:
    """
    Calculates areas of ar images detected or labeled by bounding boxes in a single file
    Try: except: here is used because of zeros in bb_img_detected after intersection_over_union sorting
    when len(bb_file_labeled) > len(bb_file_detected) 3>2
    [[bb],[bb],[bb]]
    [[bb],0,[bb]]

    Parameters:
    :param img_width: Width of an image
    :type img_width: list
    :param img_height: Height of an image
    :type img_height: list
    :return: A list of img areas for one file
    :rtype: list[float|int]
    """
    bb_file_areas = []
    bb_base = [0.5, 0.5, 1, 1]
    for bb_img in bb_file:
        bb_intersection = intersection_yolo(bb_img_base=bb_base, bb_img=bb_img)
        try:
            bb_file_areas.append(
                float(bb_intersection[3])
                * img_width
                * float(bb_intersection[4])
                * img_height
            )
        except TypeError:
            bb_file_areas.append(0)
    return bb_file_areas


def calculate_file_areas_sum(bb_areas: list[list[float | int]]) -> list[float]:
    """
    Sums areas of all images corresponding to their file. Creates a list with summed areas

    Parameters:
    :param bb_areas: A list with areas of all images in bb
    :type bb_areas: list
    :return: A list of summed areas for detected images across each file
    :rtype: list[float|int]
    """
    bb_file_areas_sum = []
    for bb_file_areas in bb_areas:
        bb_file_areas_sum.append(sum(bb_file_areas))
    return bb_file_areas_sum
