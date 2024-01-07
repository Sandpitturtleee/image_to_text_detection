import collections
import os
from pathlib import Path

import cv2

from ultralytics import YOLO

from definitions import CUT_IMAGES_DIR, RAW_IMAGES_DIR, CUT_CLASSES_IMAGES_DIR, TRAIN_DATASET_PATH


def recognize_and_crop_newspaper(model_name: str):
    model = YOLO(model_name)

    for file in os.listdir(RAW_IMAGES_DIR):
        results = model(RAW_IMAGES_DIR + file, device="mps")
        classes_names = get_cropped_images_names(model=model, results=results)
        crop_images(file=file, results=results, classes_names=classes_names, input_folder=RAW_IMAGES_DIR,
                    output_folder=CUT_IMAGES_DIR)


def recognize_and_crop_articles(model_name: str):
    model = YOLO(model_name)

    for file in os.listdir(CUT_IMAGES_DIR):
        results = model(CUT_IMAGES_DIR + file, device="mps")
        classes_names = get_cropped_images_names(model=model, results=results)
        crop_images(file=file, results=results, classes_names=classes_names, input_folder=CUT_IMAGES_DIR,
                    output_folder=CUT_CLASSES_IMAGES_DIR)

        # image = CUT_IMAGES_DIR + file
        # crop_images_X(imaage=image, file=file, results=results, classes_names=classes_names, in_folder=CUT_IMAGES_DIR,
        #               out_folder=CUT_CLASSES_IMAGES_DIR)


def train_model(dataset_path: str):

    # yolo task=detect mode=train model=yolov8n.pt
    # data=/Users/sold/Desktop/Python/Projects/image_to_text_detection/database/train/datasets/Newspaper_articles/data.yaml
    # epochs=1000 imgsz=640

    # Load a model
    model = YOLO("yolov8n.yaml")  # load a pretrained model (recommended for training)
    # Train the model with 2 GPUs
    results = model.train(
        data=TRAIN_DATASET_PATH+dataset_path,
        epochs=1000, imgsz=640, device="mps")


def crop_images(file: str, results: list, classes_names: list, input_folder: str, output_folder: str):
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
                output_folder + file_name + "_" + classes_names[i] + ".png",
                ultralytics_crop_object,
            )
        except:
            print("No classes were found for an image")


def crop_images_X(file: str, imaage: str, results: list, classes_names: list, in_folder: str, out_folder: str):
    # Load the original image

    img = cv2.imread(imaage)
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
                out_folder + file_name + "_" + classes_names[i] + ".png",
                ultralytics_crop_object,
            )
        except:
            print("No classes were found for an image")


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
