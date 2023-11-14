import os
from pathlib import Path

from demo import recognize_text

from definitions import ROOT_DIR

RAW_IMAGES_DIR = ROOT_DIR + "/database/raw_images/"
RESULTS_TXT_DIR = ROOT_DIR + "/database/results_txt/"


def save_to_txt(result: str, file: str):
    file_name = Path(file).stem
    file_path = os.path.join(RESULTS_TXT_DIR, file_name + ".txt")
    file1 = open(file_path, "w")
    file1.write(result)
    file1.close()


def detect_and_save_to_file():
    for file in os.listdir(RAW_IMAGES_DIR):
        result = recognize_text(img_path=RAW_IMAGES_DIR + file)
        save_to_txt(result=result, file=file)


if __name__ == "__main__":
    detect_and_save_to_file()
