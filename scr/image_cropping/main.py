from definitions import TRAIN_DATASET_PATH
from scr.image_cropping.functions import (object_recognition_and_cropping,
                                          train_model)

if __name__ == "__main__":
    print("OK")
    # object_recognition_and_cropping()
    train_model(dataset_path=TRAIN_DATASET_PATH)
