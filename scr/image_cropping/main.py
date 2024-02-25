import copy
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from definitions import ARTICLES_CROPPED_DIR, ARTICLES_DIR, PAGES_DIR
from scr.image_cropping.coordinate_sorting import create_cols, list_min_val
from scr.image_cropping.cropping import (
    clear_folders,
    convert_pdf_to_images,
    cropping,
    detect_and_crop_images_articles,
    detect_and_crop_images_pages,
)

if __name__ == "__main__":
    print("START")
    cropping()
