import os

import keras_ocr
import tensorflow as tf
from keras import backend as K

from definitions import RESULTS_DIR
from scr.text_recognition.organising_files import (
    create_file_paths,
    get_directory_names,
    get_file_names,
)


def create_folder_path_for_images():
    parent_dir = RESULTS_DIR
    sub_folders_n = get_directory_names(folder_path=parent_dir)
    folder_paths = []
    for sub_folder_n in sub_folders_n:
        sub_folder_n_path = os.path.join(RESULTS_DIR, sub_folder_n)
        sub_folders_p = get_directory_names(folder_path=sub_folder_n_path)
        for sub_folder_p in sub_folders_p:
            sub_folder_p_path = os.path.join(sub_folder_n_path, sub_folder_p)
            sub_folders_a = get_directory_names(folder_path=sub_folder_p_path)
            for sub_folder_a in sub_folders_a:
                sub_folder_a_path = os.path.join(sub_folder_p_path, sub_folder_a)
                folder_paths.append(sub_folder_a_path)
    return folder_paths


def create_img_path_for_images():
    folder_paths = create_folder_path_for_images()
    images_path = []
    for folder_path in folder_paths:
        folder_path = f"{folder_path}/"
        images = get_file_names(folder_path=folder_path)
        images = remove_img_files(images=images)
        images_path.append(
            create_file_paths(file_names=images, folder_path=folder_path)
        )
    return images_path


def detect_text_keras():
    images_path = create_img_path_for_images()
    images_path = flatten(xss=images_path)
    pipeline = keras_ocr.pipeline.Pipeline()

    predictions = pipeline.recognize(images_path)

    for prediction, path in zip(predictions, images_path):
        results_txt = []
        for text, box in prediction:
            results_txt.append(f"{text} ")
        detected_text = "".join(results_txt)
        save_to_txt_file(path=path, detected_text=detected_text)
    # K.clear_session()


def remove_img_files(images):
    for file in images:
        if "image" in file:
            images.remove(file)
    return images


def save_to_txt_file(path, detected_text):
    path = path[:-4] + ".txt"
    with open(path, "w") as text_file:
        text_file.write(detected_text)


def flatten(xss):
    return [x for xs in xss for x in xs]


# def detect_text():
#     pipeline = keras_ocr.pipeline.Pipeline()
#     parent_dir = RESULTS_DIR
#     sub_folders_n = get_directory_names(folder_path=parent_dir)
#     for sub_folder_n in sub_folders_n:
#         sub_folder_n_path = os.path.join(RESULTS_DIR, sub_folder_n)
#         sub_folders_p = get_directory_names(folder_path=sub_folder_n_path)
#         for sub_folder_p in sub_folders_p:
#             sub_folder_p_path = os.path.join(sub_folder_n_path, sub_folder_p)
#             sub_folders_a = get_directory_names(folder_path=sub_folder_p_path)
#             for sub_folder_a in sub_folders_a:
#                 sub_folder_a_path = os.path.join(sub_folder_p_path, sub_folder_a)
#                 print(sub_folder_a_path)
#                 detect_text_in_folder(folder_path=sub_folder_a_path,pipeline=pipeline)
#                 print()
#
# def detect_text_in_folder(folder_path,pipeline):
#
#     folder_path = f"{folder_path}/"
#     print(folder_path)
#     images = get_file_names(folder_path=folder_path)
#     images = remove_img_files(images=images)
#     images_path = create_file_paths(file_names=images,folder_path=folder_path)
#
#     predictions = pipeline.recognize(images_path)
#
#     for prediction,path in zip(predictions,images_path):
#         results_txt = []
#         for text, box in prediction:
#             results_txt.append(f"{text} ")
#         detected_text = ''.join(results_txt)
#         save_to_txt_file(path=path,detected_text=detected_text)
#     K.clear_session()
