import os
import easyocr

from definitions import RESULTS_DIR
from scr.text_recognition.organising_files import (
    create_file_paths,
    get_directory_names,
    get_file_names,
)


def detect_text_easy_ocr():
    """
    Detecting text using easy ocr
    """
    reader = easyocr.Reader(["pl"])
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
                detect_text_in_folder(folder_path=sub_folder_a_path, reader=reader)


def detect_text_in_folder(folder_path, reader):
    """
    Detecting text from images in a single folder
    """
    folder_path = f"{folder_path}/"
    images = get_file_names(folder_path=folder_path)
    images = remove_img_files(images=images)
    images_path = create_file_paths(file_names=images, folder_path=folder_path)
    body_result = []
    for path in images_path:
        result = reader.readtext(path, detail=0)
        detected_text = reformat_results(result=result)
        if "body" in path:
            body_result.append(detected_text)
        save_to_txt_file(path=path, detected_text=detected_text)
    write_body_result_file(images_path=images_path, body_result=body_result)


def remove_img_files(images: list) -> list:
    """
    remove images of a detected class "image" from images list

    Parameters:
    :param images: A list of images
    :type images: list
    :return: A list of images with removed images of "image" detected class
    :rtype: list
    """
    for file in images:
        if "image" in file:
            images.remove(file)
    return images


def save_to_txt_file(path: str, detected_text: str):
    """
    Create it and save detected text to a txt file

    Parameters:
    :param path: Path of the file
    :type path: str
    :param detected_text: Detected text
    :type detected_text: str
    """
    path = path[:-4] + ".txt"
    with open(path, "w") as text_file:
        text_file.write(detected_text)


def reformat_results(result: str) -> str:
    """
    Reformat result, join detected strings

    Parameters:
    :param result: Txt result
    :type result: str
    :return: Reformatted result
    :rtype: str
    """
    results_formatted = []
    for item in result:
        if item[-1] != "-":
            item += " "
        else:
            item = item[:-1]
        results_formatted.append(item)
    result = "".join(results_formatted)
    return result


def write_body_result_file(images_path: list, body_result: list):
    """
    Join and write results to txt file

    Parameters:
    :param images_path: Path to files
    :type images_path: list
    :param body_result: Txt result
    :type body_result: list
    """
    if len(images_path) != 0:
        path = images_path[0]
        path = path[:-4]
        path = path + "_results.txt"
        result = "".join(body_result)
        save_to_txt_file(path=path, detected_text=result)
