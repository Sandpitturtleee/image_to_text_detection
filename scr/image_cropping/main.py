from definitions import PAGES_DIR, ARTICLES_DIR, ARTICLES_CROPPED_DIR
from scr.helpers.functions import convert_pdf_to_images
from scr.image_cropping.functions import detect_and_crop_images

if __name__ == "__main__":
    print("START")
    convert_pdf_to_images(pdf_name="1.pdf", first_page=0, last_page=2)
    detect_and_crop_images(model_name="newspaper_best.pt", input_folder=PAGES_DIR, output_folder=ARTICLES_DIR)
    detect_and_crop_images(model_name="article_best.pt", input_folder=ARTICLES_DIR,
                           output_folder=ARTICLES_CROPPED_DIR)

    # TODO
    # 2) deleting old files before nex detection
