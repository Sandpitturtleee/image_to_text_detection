import copy
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from definitions import ARTICLES_CROPPED_DIR, ARTICLES_DIR, PAGES_DIR
from scr.image_cropping.coordinate_sorting import list_min_val, create_cols, bubbleSort
from scr.image_cropping.functions import (
    clear_folders,
    convert_pdf_to_images,
    detect_and_crop_images_pages,
    detect_and_crop_images_articles,
)

if __name__ == "__main__":
    print("START")
    # clear_folders()
    # convert_pdf_to_images(pdf_name="1.pdf", first_page=5, last_page=6)
    # detect_and_crop_images_pages(
    #     model_name="newspaper_best.pt",
    #     input_folder=PAGES_DIR,
    #     output_folder=ARTICLES_DIR,
    # )
    detect_and_crop_images_articles(
        model_name="article_best.pt",
        input_folder=ARTICLES_DIR,
        output_folder=ARTICLES_CROPPED_DIR,
    )
    # base_img_size = [[859, 1388]]
    # xy = [
    #     [
    #         "body",
    #         0.8470009214248035,
    #         0.44628430237344086,
    #         0.29744016138860596,
    #         0.34518355663640354,
    #     ],
    #     [
    #         "body",
    #         0.5211139541288471,
    #         0.4266076211627004,
    #         0.33282580836410436,
    #         0.47636316832616626,
    #     ],
    #     [
    #         "body",
    #         0.17432361598342344,
    #         0.7993852906680589,
    #         0.3290872140868856,
    #         0.3847988249589112,
    #     ],
    #     [
    #         "body",
    #         0.51919450859807,
    #         0.8850254201751621,
    #         0.32889554220250655,
    #         0.207435629896884,
    #     ],
    #     [
    #         "body",
    #         0.51919450859807,
    #         0.3850254201751621,
    #         0.32889554220250655,
    #         0.207435629896884,
    #     ],
    # ]
    #
    #
    # cols = create_cols(xy=xy)
    # #pprint(cols)
    # for item in cols:
    #     sorted_item = []
    #     bubbleSort(item)
    #     print(item)


    # n = np.unique(xy[:, 0])
    # cols = {i: list(xy[xy[:, 0] == i, 1]) for i in n}
    # print(cols)
    # x, y = np.array([]), np.array([])

    # for i in cols.items():
    #     x = np.append(x, [i[0]] * len(i[1]))
    #     y = np.append(y, (i[1]))
    # plt.scatter(x, y)
    # plt.show()
