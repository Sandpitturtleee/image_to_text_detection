import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Detection
CUT_IMAGES_DIR = ROOT_DIR + "/database/detect/cut_images/"
RAW_IMAGES_DIR = ROOT_DIR + "/database/detect/raw_images/"
RESULTS_TXT_DIR = ROOT_DIR + "/database/detect/results_txt/"

# Training
TRAIN_DATASET_PATH = ROOT_DIR + "/database/train/datasets/"
TRAIN_PDF_DIR = ROOT_DIR + "/database/train/raw_files/pdf/"
TRAIN_IMAGES_DIR = ROOT_DIR + "/database/train/raw_files/images/"
