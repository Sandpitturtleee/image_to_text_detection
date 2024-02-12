import os
import shutil
import re
from os import listdir
from os.path import isfile, join
from pprint import pprint

from definitions import DETECTED_TEXT, EXPECTED_TEXT


def analyze_text():
    print()

def analyze_length():
    text_detected = read_files_detected(path=DETECTED_TEXT)
    text_expected = read_files_detected(path=EXPECTED_TEXT)
    text_length = compare_length(detected=text_detected,expected=text_expected)
    print(text_length)

def compare_length(detected,expected):
    text_length = []
    for d,e in zip(detected,expected):
        text_length.append(f"{len(e)},{len(d)}")
    return text_length

def read_files_detected(path):
    file_names_detected = get_file_names(folder_path=path)
    file_paths_detected = create_file_paths(file_names=file_names_detected,folder_path=path)
    text = []
    for file in file_paths_detected:
        text.append(read_txt_to_str(path=file))
    return text

def read_txt_to_str(path):
    with open(path, 'r') as file:
        data = file.read().rstrip()
    return data

def get_file_names(folder_path: str):
    try:
        only_files = [f for f in sorted_alphanumeric(listdir(folder_path)) if isfile(join(folder_path, f))]
        only_files.remove(".DS_Store")
    except ValueError:
        print(".Ds_Store not found")
    return only_files

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def create_file_paths(file_names,folder_path):
    file_paths = []
    for item in file_names:
        file_paths.append(f"{folder_path+item}")
    return file_paths