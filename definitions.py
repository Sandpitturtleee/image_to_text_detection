import os

"""
    File used for storing in one place global variables for easier accessibility and better order
"""
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Detection
CUT_IMAGES_DIR = ROOT_DIR + "/database/detect/cut_images/"
RAW_IMAGES_DIR = ROOT_DIR + "/database/detect/raw_images/"
CUT_CLASSES_IMAGES_DIR = ROOT_DIR + "/database/detect/cut_classes_images/"
RESULTS_TXT_DIR = ROOT_DIR + "/database/detect/results_txt/"

# Training
POPPLER_PATH = "/opt/homebrew/opt/poppler/bin"

TRAIN_DATASET_PATH = ROOT_DIR + "/database/train/datasets/"
TRAIN_PDF_DIR = ROOT_DIR + "/database/train/raw_files/pdf/"
TRAIN_IMAGES_DIR = ROOT_DIR + "/database/train/raw_files/images/"
