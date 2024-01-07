import os
from pathlib import Path

import easyocr
from demo import recognize_text
from matplotlib import pyplot as plt

from definitions import ARTICLES_DIR, RESULTS_DIR


def save_to_txt(result: str, file: str):
    file_name = Path(file).stem
    file_path = os.path.join(RESULTS_DIR, file_name + ".txt")
    file1 = open(file_path, "w")
    file1.write(result)
    file1.close()


def detect_and_save_to_file():
    for file in os.listdir(ARTICLES_DIR):
        result = recognize_text(img_path=ARTICLES_DIR + file)
        save_to_txt(result=result, file=file)


import keras_ocr

if __name__ == "__main__":
    print()
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
