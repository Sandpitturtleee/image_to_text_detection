from definitions import TRAIN_DATASET_PATH, RAW_IMAGES_DIR, CUT_IMAGES_DIR, CUT_CLASSES_IMAGES_DIR
from scr.helpers.functions import convert_pdf_to_images
from scr.image_cropping.functions import detect_and_crop_images

if __name__ == "__main__":
    print("START")
    convert_pdf_to_images(pdf_name="1.pdf", first_page=0, last_page=2)
    detect_and_crop_images(model_name="newspaper_best.pt", input_folder=RAW_IMAGES_DIR, output_folder=CUT_IMAGES_DIR)
    detect_and_crop_images(model_name="article_best.pt", input_folder=CUT_IMAGES_DIR,
                           output_folder=CUT_CLASSES_IMAGES_DIR)

    # TODO
    # 1) raw_files to correct folder, rename folder to good names
    # 2) deleting old files before nex detection
