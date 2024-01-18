from definitions import ARTICLES_CROPPED_DIR, ARTICLES_DIR, PAGES_DIR
from scr.image_cropping.functions import (clear_folders, convert_pdf_to_images,
                                          detect_and_crop_images_pages, detect_and_crop_images_articles)

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
