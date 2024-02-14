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

    text_expected = read_files_detected(path=EXPECTED_TEXT)
    text_detected = read_files_detected(path=DETECTED_TEXT)

    # text_expected =["Doniesienia o kibolach bijących piłkarzy Legii, który ośmielili się przegrać z Lechem, szokują tylko niezorientowanych w realiach tzw. ekstraklasy."]
    # text_detected =["oniesienia 0 kibolach D bijących piłkarzy Lcgii; którzy ośmielili SIę przegrać z Lechem; szokują tyl ko niczoricntowanych w rcaliach tzw. ekstraklasy."]
    converted_detected = create_list_of_words(text=text_detected)
    converted_expected = create_list_of_words(text=text_expected)

    article_percentages = word_matching(text_expected=converted_expected, text_detected=converted_detected, word_threshold=1,search_depth=1000000)
    print(article_percentages)
    # text_length = compare_length(detected=text_detected,expected=text_expected)
    # print(text_length)

def word_matching(text_expected,text_detected,word_threshold,search_depth):
    article_percentages = []
    for e,d in zip(text_expected,text_detected):
        article_percentage = word_matching_article(expected=e,detected=d,word_threshold=word_threshold,search_depth=search_depth)
        article_percentages.append(article_percentage)
    return article_percentages

def word_matching_article(expected,detected,word_threshold,search_depth):
    matching_words = 0
    for i in range(len(expected)):
        depth = 0
        try:
            for j in range(len(detected)):
                percentage = match_word_percentage(e=expected[i],d=detected[j+i])
                depth += 1
                if percentage >= word_threshold:
                    matching_words += 1
                if depth > search_depth:
                    break
        except IndexError:
            pass
    return matching_words/len(expected)

def match_word_percentage(e,d):
    matching = 0
    e,d = add_string_parts(e=e,d=d)
    for item_e,item_d in zip(e,d):
        if item_e == item_d:
            matching +=1
    return matching/len(e)


def add_string_parts(e,d):
    e_length = len(e)
    d_length = len(d)
    length_diff = abs(e_length - d_length)
    string_diff = "#" * length_diff
    if e_length < d_length:
        e += string_diff
    elif e_length > d_length:
        d += string_diff
    return e,d
def create_list_of_words(text):
    result = []
    for item in text:
        result.append(list(item.split(" ")))
    return result
    # expected_result = list(expected.split(" "))
    # return expected_result

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