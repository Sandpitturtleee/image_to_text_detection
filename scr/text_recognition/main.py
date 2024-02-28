from scr.text_recognition.organising_files import (
    creating_folder_structure,
)

from scr.text_recognition.text_detection_easyocr import (
    detect_text_easy_ocr,
)

if __name__ == "__main__":
    print()
    creating_folder_structure()
    detect_text_easy_ocr()
