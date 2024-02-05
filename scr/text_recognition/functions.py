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
