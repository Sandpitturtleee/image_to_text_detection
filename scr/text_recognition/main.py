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
    create_articles_folders, move_files_to_folders, create_results_txt_articles_folder, organizing,
)

if __name__ == "__main__":
    print()
    organizing()

    ########################### BIGGER IMAGE SIZE FOR TRAINING

    # pipeline = keras_ocr.pipeline.Pipeline()
    # images = [
    #     keras_ocr.tools.read(img) for img in [
    #         '/Users/sold/Desktop/Python/Projects/image_to_text_detection/database/detect/4_articles_cropped/1_page_2_article_1_body_5.png',
    #     ]
    # ]
    # # Predictions is a list of (string, box) tuples.
    # predictions = pipeline.recognize(images)
    #
    # predicted_image_1 = predictions[0]
    # results_txt = []
    # for text, box in predicted_image_1:
    #     results_txt.append(f"{text} ")
    # detected_text = ''.join(results_txt)
    # print(detected_text)



    # reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory
    # result = reader.readtext('/Users/sold/Desktop/Python/Projects/image_to_text_detection/database/3_articles/demo1.png')
    # for item in result:
    #     print(item[1])
