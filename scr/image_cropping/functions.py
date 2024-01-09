import collections
import os
import shutil
from pathlib import Path

import cv2
from PIL import Image
from pdf2image import convert_from_path
from ultralytics import YOLO

from definitions import (
    ARTICLES_CROPPED_DIR,
    ARTICLES_DIR,
    NEWSPAPERS_DIR,
    PAGES_DIR,
    POPPLER_PATH,
    RESULTS_DIR,
)

# pip install pdf2image
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# brew install poppler
# brew --prefix poppler
# pop_path = "/opt/homebrew/opt/poppler/bin"


"""
    A file containing functions that are used to detect objects on images an crop them into new images
"""


def detect_and_crop_images(model_name: str, input_folder: str, output_folder: str):
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
        crop_image(
            file=file,
            results=results,
            detected_classes=detected_classes,
            input_folder=input_folder,
            output_folder=output_folder,
        )


def crop_image(
    file: str,
    results: list,
    detected_classes: list,
    input_folder: str,
    output_folder: str,
):
    """
    Crops and saves an images with regard to detected bounding boxes.

    Parameters:
    :param file: Input img file on which cropping is performed
    :type file: str
    :param results: Detected bounding boxes
    :type results: list
    :param detected_classes: Names of classes used for naming for new images
    :type detected_classes: list
    :param input_folder: Input folder for images to recognise classes
    :type input_folder: str
    :param output_folder: Output folder for saving cropped images
    :type output_folder: str
    """

    # # Showing image
    # result = results[0]
    # for box in result.boxes:
    #     label = result.names[box.cls[0].item()]
    #     cords = [round(x) for x in box.xyxy[0].tolist()]
    #     prob = box.conf[0].item()
    # im = Image.fromarray(result.plot()[:, :, ::-1])
    # im.show()

    # Load the original image
    image = input_folder + file
    img = cv2.imread(image)
    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.tolist()

    # Iterate through the bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        # Crop the object using the bounding box coordinates
        ultralytics_crop_object = img[int(y1) : int(y2), int(x1) : int(x2)]
        # Save the cropped object as an image
        file_name = Path(file).stem
        try:
            cv2.imwrite(
                output_folder + file_name + "_" + detected_classes[i] + ".png",
                ultralytics_crop_object,
            )
        except:
            print("No classes were found for an image")


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
    pages = convert_from_path(
        NEWSPAPERS_DIR + pdf_name,
        poppler_path=POPPLER_PATH,
        first_page=first_page,
        last_page=last_page,
    )
    pdf_name = Path(NEWSPAPERS_DIR + pdf_name).stem
    for i in range(len(pages)):
        pages[i].save(f"{PAGES_DIR}{pdf_name}_page_{str(i)}.png", "PNG")


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
