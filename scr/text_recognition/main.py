import os
from pathlib import Path

from demo import recognize_text

from definitions import RESULTS_TXT_DIR, CUT_IMAGES_DIR


def save_to_txt(result: str, file: str):
    file_name = Path(file).stem
    file_path = os.path.join(RESULTS_TXT_DIR, file_name + ".txt")
    file1 = open(file_path, "w")
    file1.write(result)
    file1.close()


def detect_and_save_to_file():
    for file in os.listdir(CUT_IMAGES_DIR):
        result = recognize_text(img_path=CUT_IMAGES_DIR + file)
        save_to_txt(result=result, file=file)


if __name__ == "__main__":
    detect_and_save_to_file()
