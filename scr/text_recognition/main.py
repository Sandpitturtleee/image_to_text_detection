from demo import recognize_text

from definitions import ROOT_DIR

RAW_IMAGES_DIR = ROOT_DIR + "/database/raw_images/"

if __name__ == "__main__":
    print(recognize_text(img_path=RAW_IMAGES_DIR + "demo.png"))
