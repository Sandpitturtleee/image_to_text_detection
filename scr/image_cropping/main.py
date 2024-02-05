import copy
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from definitions import ARTICLES_CROPPED_DIR, ARTICLES_DIR, PAGES_DIR
from scr.image_cropping.coordinate_sorting import list_min_val, create_cols
from scr.image_cropping.functions import (
    clear_folders,
    convert_pdf_to_images,
    detect_and_crop_images_pages,
    detect_and_crop_images_articles,
    cropping,
)

if __name__ == "__main__":
    print("START")
    cropping()
