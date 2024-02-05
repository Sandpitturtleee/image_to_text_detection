import os
from pathlib import Path

import easyocr
from demo import recognize_text
from matplotlib import pyplot as plt

from definitions import ARTICLES_DIR, RESULTS_DIR


import keras_ocr

from scr.text_recognition.organising_files import (
    create_newspaper_folders,
    create_pages_folders,
    create_articles_folders,
)

if __name__ == "__main__":
    print()
    # create_newspaper_folders()
    # create_pages_folders()
    create_articles_folders()

    ############################ BIGGER IMAGE SIZE FOR TRAINING
    # detect_and_save_to_file()

    # pipeline = keras_ocr.pipeline.Pipeline()
    # images = [
    #     keras_ocr.tools.read(img) for img in [
    #         '/Users/sold/Desktop/Python/Projects/image_to_text_detection/database/3_articles/demo1.png',
    #     ]
    # ]
    # # Predictions is a list of (string, box) tuples.
    # predictions = pipeline.recognize(images)
    #
    # predicted_image_1 = predictions[0]
    # for text, box in predicted_image_1:
    #     print(text)

    # reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory
    # result = reader.readtext('/Users/sold/Desktop/Python/Projects/image_to_text_detection/database/3_articles/demo1.png')
    # for item in result:
    #     print(item[1])
