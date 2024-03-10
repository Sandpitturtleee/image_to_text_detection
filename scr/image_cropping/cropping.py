import collections
import os
import shutil
from pathlib import Path

import cv2
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from PIL import Image
from ultralytics import YOLO

from definitions import (
    ARTICLES_CROPPED_DIR,
    ARTICLES_DIR,
    NEWSPAPERS_CROP_DIR,
    NEWSPAPERS_DIR,
    PAGES_DIR,
    POPPLER_PATH,
    RESULTS_DIR,
)
from scr.image_cropping.coordinate_sorting import (
    convert_bb_file_voc_to_yolo,
    get_img_sizes,
    sort_body_elements_in_article,
)
from scr.image_cropping.helpers import add_img_names_to_boxes, divide_bb_file_detected

# pip install pdf2image
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# brew install poppler
# brew --prefix poppler
# pop_path = "/opt/homebrew/opt/poppler/bin"


"""
    A file containing functions that are used to detect objects on images an crop them into new images
"""


def cropping():
    """
    Grouping function responsible for looping over all files in NEWSPAPERS_CROP_DIR and detecting images and
    cropping them to correct folders. It also removes all previous files in used directories to ensure that
    everything works as intended.
    """
    clear_folders()
    for file in os.listdir(NEWSPAPERS_CROP_DIR):
        convert_pdf_to_images(pdf_name=file, first_page=0, last_page=5)
        detect_and_crop_images_pages(
            model_name="newspaper_best.pt",
            input_folder=PAGES_DIR,
            output_folder=ARTICLES_DIR,
        )
        detect_and_crop_images_articles(
            model_name="article_best.pt",
            input_folder=ARTICLES_DIR,
            output_folder=ARTICLES_CROPPED_DIR,
        )


def detect_and_crop_images_pages(
    model_name: str, input_folder: str, output_folder: str
):
    """
    Detect classes images and crop them to create new img files.

    Parameters:
    :param model_name: Model to use for object detection
    :type model_name: str
    :param input_folder: Input folder for images to recognise classes
    :type input_folder: str
    :param output_folder: Output folder for saving cropped images
    :type output_folder: str
    """
    model = YOLO(model_name)
    for file in os.listdir(input_folder):
        results = model(input_folder + file, device="mps")
        detected_classes = get_cropped_images_names(model=model, results=results)
        box = results[0].boxes.xyxy.tolist()
        crop_image(
            file=file,
            boxes=box,
            detected_classes=detected_classes,
            input_folder=input_folder,
            output_folder=output_folder,
        )


def detect_and_crop_images_articles(
    model_name: str, input_folder: str, output_folder: str
):
    """
    Detect classes images and crop them to create new img files.

    Parameters:
    :param model_name: Model to use for object detection
    :type model_name: str
    :param input_folder: Input folder for images to recognise classes
    :type input_folder: str
    :param output_folder: Output folder for saving cropped images
    :type output_folder: str
    """
    model = YOLO(model_name)
    names = model.names
    img_sizes = get_img_sizes(
        "/Users/sold/Desktop/Python/Projects/image_to_text_detection/database/detect/3_articles/"
    )
    iterate = 0
    for file in os.listdir(input_folder):
        results = model(input_folder + file, device="mps")
        box = results[0].boxes.xyxy.tolist()

        bb_file_detected = add_img_names_to_boxes(
            names=names, results=results, bb_labeled=box
        )

        bb_file_detected_body, bb_file_detected_other = divide_bb_file_detected(
            bb_file_detected=bb_file_detected
        )

        classes_names_other = number_classes_names_other(
            bb_file_detected_other=bb_file_detected_other
        )
        bb_file_detected_other = remove_classes_names_from_bb_file(
            bb_file_detected=bb_file_detected_other
        )

        bb_file_detected_body = convert_bb_file_voc_to_yolo(
            bb_file_voc=bb_file_detected_body,
            img_width=img_sizes[iterate][0],
            img_height=img_sizes[iterate][1],
        )
        classes_names_body = create_names_for_body(
            bb_file_detected_body=bb_file_detected_body
        )

        if len(bb_file_detected_body) != 0:
            bb_file_detected_body = sort_body_elements_in_article(
                bb_file_detected_body=bb_file_detected_body
            )

            bb_file_detected_body = remove_classes_names_from_bb_file(
                bb_file_detected=bb_file_detected_body
            )
            bb_file_detected_body = convert_yolo_to_coco(
                bb_file_detected_body=bb_file_detected_body,
                img_width=img_sizes[iterate][0],
                img_height=img_sizes[iterate][1],
            )

        crop_image(
            file=file,
            boxes=bb_file_detected_other,
            detected_classes=classes_names_other,
            input_folder=input_folder,
            output_folder=output_folder,
        )
        crop_image(
            file=file,
            boxes=bb_file_detected_body,
            detected_classes=classes_names_body,
            input_folder=input_folder,
            output_folder=output_folder,
        )
        iterate += 1


def crop_image(
    file: str,
    boxes: list,
    detected_classes: list,
    input_folder: str,
    output_folder: str,
):
    """
    Crops and saves an images with regard to detected bounding boxes.

    Parameters:
    :param file: Input img file on which cropping is performed
    :type file: str
    :param boxes: Detected bounding boxes
    :type boxes: list
    :param detected_classes: Names of classes used for naming for new images
    :type detected_classes: list
    :param input_folder: Input folder for images to recognise classes
    :type input_folder: str
    :param output_folder: Output folder for saving cropped images
    :type output_folder: str
    """

    # Load the original image
    image = input_folder + file
    img = cv2.imread(image)
    # Extract bounding boxes

    # Iterate through the bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        ultralytics_crop_object = img[int(y1) : int(y2), int(x1) : int(x2)]
        file_name = Path(file).stem
        try:
            cv2.imwrite(
                output_folder + file_name + "_" + detected_classes[i] + ".png",
                ultralytics_crop_object,
            )
        except:
            print("No classes were found for an image")


def number_classes_names_other(
    bb_file_detected_other: list[list[list[float | str]]],
) -> list[str]:
    """
    Takes names of cropped files that are not "body" type images and names them correctly for file creation.

    Parameters:
    :param bb_file_detected_other: List of images with its name and bounding boxes
    :type bb_file_detected_other: list[list[float | str]]
    :return: A list of names for new files
    :rtype: list[str]
    """
    bb_file_detected_names_retrieved = get_detected_classes_name_from_bb_file(
        bb_file_detected_other=bb_file_detected_other
    )
    bb_file_detected_other_names = numbering_classes_names(
        detected_classes=bb_file_detected_names_retrieved
    )
    return bb_file_detected_other_names


def remove_classes_names_from_bb_file(
    bb_file_detected: list[list[list[float | str]]],
) -> list[float | int]:
    """
    Takes images with its classes names and bounding boxes and removes names.

    Parameters:
    :param bb_file_detected: List of images with its name and bounding boxes
    :type bb_file_detected: list[list[list[float | str]]]
    :return: A list of images with its bounding boxes without names
    :rtype: list[float | int]
    """
    for bb_img_detected in bb_file_detected:
        bb_img_detected.pop(0)
    return bb_file_detected


def get_detected_classes_name_from_bb_file(
    bb_file_detected_other: list[list[list[float | str]]],
) -> list[list[float | str]]:
    """
    Takes names of cropped files from list with bounding boxes and creates a list with only names of the classes.

    Parameters:
    :param bb_file_detected_other: List of images with its name and bounding boxes
    :type bb_file_detected_other: list[list[list[float | str]]]
    :return: A list of names of detected classes
    :rtype: list[list[float | str]]
    """
    bb_img_detected_names_retrieved = []
    for bb_img_detected in bb_file_detected_other:
        bb_img_detected_names_retrieved.append(bb_img_detected[0])
    return bb_img_detected_names_retrieved


def get_cropped_images_names(model: YOLO, results: list) -> list:
    """
    Creates a list of names for naming new files.

    Parameters:
    :param model: YOLO model used for object detection
    :type model: YOLO
    :param results: Detected bounding boxes
    :type results: list
    :return: A list of images classes names detected by YOLO
    :rtype: List[str]
    """
    names = model.names
    detected_classes = []
    for r in results:
        for c in r.boxes.cls:
            detected_classes.append(names[int(c)])
    detected_classes = numbering_classes_names(detected_classes=detected_classes)

    return detected_classes


def numbering_classes_names(detected_classes: list) -> list:
    """
    Crops and saves an images with regard to detected bounding boxes.

    Parameters:
    :param detected_classes: A list of images classes names detected by YOLO
    :type detected_classes: List[str]
    :return: A list of images classes names detected by YOLO with added numbering
    :rtype: List[str]
    """
    duplicates = [
        item
        for item, count in collections.Counter(detected_classes).items()
        if count > 1
    ]
    unique = [
        item
        for item, count in collections.Counter(detected_classes).items()
        if count == 1
    ]

    # Changing unique names in a list
    for i in range(len(detected_classes)):
        if detected_classes[i] in unique:
            detected_classes[i] += f"_{1}"

    # Changing duplicate names in a list
    for j in range(len(duplicates)):
        duplicate_number = 1
        for i in range(len(detected_classes)):
            if detected_classes[i] == duplicates[j]:
                detected_classes[i] += f"_{duplicate_number}"
                duplicate_number += 1

    return detected_classes


def convert_pdf_to_images(pdf_name: str, first_page: int, last_page: int):
    """
    Converts a document in a pdf format to multiple png images corresponding to 2_pages

    Parameters:
    :param pdf_name: Name to the PDF that you want to convert
    :type pdf_name: str
    :param first_page: First page of a PDF file from where you want to start converting.
    :type first_page: int
    :param last_page: Last page of a PDF file to where you want to start converting.
    :type last_page: int
    """
    try:
        pages = convert_from_path(
            NEWSPAPERS_DIR + pdf_name,
            poppler_path=POPPLER_PATH,
            first_page=first_page,
            last_page=last_page,
        )
        pdf_name = Path(NEWSPAPERS_DIR + pdf_name).stem
        for i in range(len(pages)):
            pages[i].save(f"{PAGES_DIR}{pdf_name}_page_{str(i)}.png", "PNG")
    except PDFPageCountError:
        print(".DS_Store")


def clear_folders():
    """
    Clearing folders from files before every detection and cropping
    """
    folders = [PAGES_DIR, ARTICLES_DIR, ARTICLES_CROPPED_DIR, RESULTS_DIR]
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


def create_names_for_body(
    bb_file_detected_body: list[list[float | int]],
) -> list[str]:
    """
    Creates names for body image files with correct numbering

    Parameters:
    :bb_file_detected_body: List of detected body images
    :type bb_file_detected_body: list[list[float | int]]
    :return: A list of names for new files with added numbers
    :rtype: list[str]
    """
    names = []
    for i in range(len(bb_file_detected_body)):
        names.append("body_" + str(i + 1))
    return names


def convert_yolo_to_coco(
    bb_file_detected_body: list[list[list[float | str]]],
    img_width: int,
    img_height: int,
) -> list[list[list[float | str]]]:
    """
        Creates bounding boxes format from yolo to coco style

        Parameters:
        :bb_file_detected_body: List of detected body images
        :type bb_file_detected_body: list[list[list[float | str]]]
        :img_width: List of detected body images
        :type img_width: int
        :img_height: List of detected body images
        :type img_height: int
    :return: List of detected body images converted to coco format
    :rtype: list[str]
    """
    converted = []
    for item in bb_file_detected_body:
        center_x, center_y, width, height = item
        x1 = (center_x - width / 2) * img_width
        x2 = (center_x + width / 2) * img_width
        y1 = (center_y - height / 2) * img_height
        y2 = (center_y + height / 2) * img_height
        converted.append([x1, y1, x2, y2])
    return converted
