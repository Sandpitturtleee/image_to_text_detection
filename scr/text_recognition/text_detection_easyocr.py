import os

import keras_ocr
from keras import backend as K
import tensorflow as tf
import easyocr



from definitions import RESULTS_DIR
from scr.text_recognition.organising_files import get_file_names, create_file_paths, get_directory_names

def detect_text_easy_ocr():
    reader = easyocr.Reader(['pl'])
    parent_dir = RESULTS_DIR
    sub_folders_n = get_directory_names(folder_path=parent_dir)
    for sub_folder_n in sub_folders_n:
        sub_folder_n_path = os.path.join(RESULTS_DIR, sub_folder_n)
        sub_folders_p = get_directory_names(folder_path=sub_folder_n_path)
        for sub_folder_p in sub_folders_p:
            sub_folder_p_path = os.path.join(sub_folder_n_path, sub_folder_p)
            sub_folders_a = get_directory_names(folder_path=sub_folder_p_path)
            for sub_folder_a in sub_folders_a:
                sub_folder_a_path = os.path.join(sub_folder_p_path, sub_folder_a)
                detect_text_in_folder(folder_path=sub_folder_a_path,reader=reader)

def detect_text_in_folder(folder_path,reader):

    folder_path = f"{folder_path}/"
    images = get_file_names(folder_path=folder_path)
    images = remove_img_files(images=images)
    images_path = create_file_paths(file_names=images,folder_path=folder_path)

    for path in images_path:
        result = reader.readtext(path, detail=0)
        detected_text = reformat_results(result=result)
        save_to_txt_file(path=path,detected_text=detected_text)



def remove_img_files(images):
    for file in images:
        if "image" in file:
            images.remove(file)
    return images

def save_to_txt_file(path,detected_text):
    path = path[:-4] + ".txt"
    with open(path, "w") as text_file:
        text_file.write(detected_text)

def flatten(xss):
    return [x for xs in xss for x in xs]


def reformat_results(result):
    results_formatted = []
    for item in result:
        # if item[-1] != "-":
        #     item += " "
        # else:
        #     item = item[:-1]
        if item[-1] == "-":
            item = item[:-1]
        results_formatted.append(item)
    result = ''.join(results_formatted)
    return result
