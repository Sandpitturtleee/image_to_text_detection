import collections
import os
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from definitions import CUT_IMAGES_DIR, RAW_IMAGES_DIR


def object_recognition_and_cropping():
    model = YOLO("yolov8n.pt")
    for file in os.listdir(RAW_IMAGES_DIR):
        results = model(RAW_IMAGES_DIR + file, device="mps")
        classes_names = get_cropped_images_names(model=model, results=results)
        crop_images(file=file, results=results, classes_names=classes_names)


def train_model(dataset_path: str):
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model with 2 GPUs
    results = model.train(data=dataset_path, epochs=100, imgsz=640, device="mps")


def crop_images(file: str, results: list, classes_names: list):
    # Load the original image
    image = RAW_IMAGES_DIR + file
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
        cv2.imwrite(
            CUT_IMAGES_DIR + file_name + "_" + classes_names[i] + ".png",
            ultralytics_crop_object,
        )


def get_cropped_images_names(model: YOLO, results: list) -> list:
    names = model.names
    classes_names = []
    for r in results:
        for c in r.boxes.cls:
            classes_names.append(names[int(c)])
    classes_names = number_classes_names(classes_names=classes_names)
    return classes_names


def number_classes_names(classes_names: list) -> list:
    duplicates = [
        item for item, count in collections.Counter(classes_names).items() if count > 1
    ]
    unique = [
        item for item, count in collections.Counter(classes_names).items() if count == 1
    ]

    # Changing unique names in a list
    for i in range(len(classes_names)):
        if classes_names[i] in unique:
            classes_names[i] += f"_{1}"

    # Changing duplicate names in a list
    for j in range(len(duplicates)):
        duplicate_number = 1
        for i in range(len(classes_names)):
            if classes_names[i] == duplicates[j]:
                classes_names[i] += f"_{duplicate_number}"
                duplicate_number += 1

    return classes_names
