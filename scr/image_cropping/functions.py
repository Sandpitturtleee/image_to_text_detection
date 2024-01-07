import collections
import os
from pathlib import Path

import cv2

from ultralytics import YOLO

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
        crop_image(file=file, results=results, detected_classes=detected_classes, input_folder=input_folder,
                   output_folder=output_folder)


def crop_image(file: str, results: list, detected_classes: list, input_folder: str, output_folder: str):
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
    # Load the original image
    image = input_folder + file
    img = cv2.imread(image)
    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.tolist()

    # Iterate through the bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        # Crop the object using the bounding box coordinates
        ultralytics_crop_object = img[int(y1): int(y2), int(x1): int(x2)]
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
        item for item, count in collections.Counter(detected_classes).items() if count > 1
    ]
    unique = [
        item for item, count in collections.Counter(detected_classes).items() if count == 1
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
