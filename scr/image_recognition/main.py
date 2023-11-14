import os

from ultralytics import YOLO
import cv2
from pathlib import Path

from definitions import RAW_IMAGES_DIR, CUT_IMAGES_DIR

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    for file in os.listdir(RAW_IMAGES_DIR):
        results = model(RAW_IMAGES_DIR + file)

        # Load the original image
        image = RAW_IMAGES_DIR + file
        img = cv2.imread(image)

        # Extract bounding boxes
        boxes = results[0].boxes.xyxy.tolist()

        # Iterate through the bounding boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Crop the object using the bounding box coordinates
            ultralytics_crop_object = img[int(y1):int(y2), int(x1):int(x2)]
            # Save the cropped object as an image
            file_name = Path(file).stem
            cv2.imwrite(CUT_IMAGES_DIR+file_name + 'crop_' + str(i) + '.png', ultralytics_crop_object)

