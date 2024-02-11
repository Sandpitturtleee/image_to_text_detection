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
from scr.text_recognition.text_correction_symspell import word_segmentation_symspell, word_correction_symspell, \
    word_segmentation_symspell1
from scr.text_recognition.text_detection_easyocr import detect_text_easy_ocr, reformat_results
from scr.text_recognition.text_detection_keras import detect_text_keras

if __name__ == "__main__":
    print()
    organizing()
    # # detect_text_keras()
    # #detect_text_in_folder()
    detect_text_easy_ocr()

    # path = "/Users/sold/Desktop/Python/Projects/image_to_text_detection/database/detect/5_results/1_newspaper/2_page/2_article/1_page_2_article_2_body_2.png"
    # reader = easyocr.Reader(['pl'])  # this needs to run only once to load the model into memory
    # result = reader.readtext(path,detail=0)
    # print(result)
    # detected_text = reformat_results(result=result)
    # print(detected_text)
    # word_segmentation_symspell1(input_term=detected_text)
    # # word_correction_symspell(input_term=detected_text)
    # # detected_text = detected_text.replace(" ", "")
    #
    # # print(detected_text)
    # # word_segmentation_symspell(input_term=detected_text)
    # # for item in result:
    # #     print(item[1])
