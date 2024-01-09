import os

"""
    File used for storing in one place global variables for easier accessibility and better order
"""
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Detection
NEWSPAPERS_DIR = ROOT_DIR + "/database/detect/1_newspapers/"
PAGES_DIR = ROOT_DIR + "/database/detect/2_pages/"
ARTICLES_DIR = ROOT_DIR + "/database/detect/3_articles/"
ARTICLES_CROPPED_DIR = ROOT_DIR + "/database/detect/4_articles_cropped/"
RESULTS_DIR = ROOT_DIR + "/database/detect/5_results/"

# Training
POPPLER_PATH = "/opt/homebrew/opt/poppler/bin"

TRAIN_DATASET_PATH = ROOT_DIR + "/database/train/datasets/"
TRAIN_IMAGES_DIR = ROOT_DIR + "/database/train/1_newspapers/images/"

PAGES_ANALYZE_IMAGES_DIR = ROOT_DIR + "/database/analyze/Newspaper/train/images/"
PAGES_ANALYZE_LABELS_DIR = ROOT_DIR + "/database/analyze/Newspaper/train/labels/"
ARTICLES_ANALYZE_IMAGES_DIR = ROOT_DIR + "/database/analyze/Newspaper_articles/train/images/"
ARTICLES_ANALYZE_LABELS_DIR = ROOT_DIR + "/database/analyze/Newspaper_articles/train/labels/"
