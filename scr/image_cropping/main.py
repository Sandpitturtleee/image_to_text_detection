from definitions import TRAIN_DATASET_PATH
from scr.helpers.functions import convert_pdf_to_images
from scr.image_cropping.functions import recognize_and_crop_newspaper, recognize_and_crop_articles

if __name__ == "__main__":
    print("OK")
    convert_pdf_to_images(pdf_name="1.pdf", first_page=0, last_page=25)
    recognize_and_crop_newspaper(model_name="newspaper_best.pt")
    recognize_and_crop_articles(model_name="article_best.pt")

    #train_model(dataset_path=TRAIN_DATASET_PATH)


