import os
import re
import shutil
from os import listdir
from os.path import isfile, join
from pprint import pprint

from scipy.spatial.distance import hamming

from definitions import DETECTED_TEXT, EXPECTED_TEXT
from scr.statistical_analysys.plot_functions import create_bar_plot


def analyze_text():
    article_percentages, percentages_list = analyze_percentages()
    print(article_percentages)
    for item in percentages_list:
        create_bar_plot(data=item, bin_edges=area_bin_edges, bin_labels=area_bin_labels)


def analyze_percentages():
    text_expected = read_files_detected(path=EXPECTED_TEXT)
    text_detected = read_files_detected(path=DETECTED_TEXT)

    # text_expected =["Doniesienia o kibolach bijących piłkarzy Legii, którzy ośmielili się przegrać z Lechem, szokują tylko niezorientowanych w realiach tzw. ekstraklasy."]
    # text_detected =["oniesienia 0 kibolach D bijących piłkarzy Lcgii; którzy ośmielili SIę przegrać z Lechem; szokują tyl ko niczoricntowanych w rcaliach tzw. ekstraklasy."]
    # text_expected = ["Doniesienia o kibolach bijących piłkarzy Legii, którzy ośmielili się przegrać"]
    # text_detected = ["oniesienia 0 0 0 0 0 0 0 kibolach D bijących piłkarzy Lcgii; którzy ośmielili SIę przegrać"]
    # text_expected = ["Dziś pierwsze urodziny tego symbolu kobiecej siły i siostrzeństwa ponad podziałami - ubiegłego 3 października powstrzymałyśmy rząd przed wprowadzeniem całkowitego zakazu aborcji."]
    # text_detected = ["ziśpierwszeurodzinyte D go symbolukobicccj siły isiostrzenstwa ponadpodziałami -ubiegłego 3 października powstrzymałyśmy rządprzcd wprowadzeniem całkowitego zakazu aborcji."]

    converted_detected = create_list_of_words(text=text_detected)
    converted_expected = create_list_of_words(text=text_expected)

    article_percentages, percentages_list = word_matching(
        text_expected=converted_expected,
        text_detected=converted_detected,
        word_threshold=0.75,
        search_depth=10,
    )
    return article_percentages, percentages_list


def word_matching(text_expected, text_detected, word_threshold, search_depth):
    article_percentages = []
    percentages_list = []
    for e, d in zip(text_expected, text_detected):
        article_percentage, percentages = word_matching_article(
            expected=e,
            detected=d,
            word_threshold=word_threshold,
            search_depth=search_depth,
            min_word_lenght=4,
        )
        article_percentages.append(article_percentage)
        percentages_list.append(percentages)
    return article_percentages, percentages_list


def word_matching_article(
    expected, detected, word_threshold, search_depth, min_word_lenght
):
    not_detected_streak, matching_words, last_word_index = 0, 0, 0
    search_depth_original = search_depth
    percentages = []
    for i in range(len(expected)):
        depth = 0
        percentages_nested = []
        try:
            for j in range(len(detected)):
                percentage = match_word_percentage(
                    e=expected[i], d=detected[j + last_word_index]
                )
                percentages_nested.append(percentage)
                depth += 1
                not_detected_streak += 1
                if not_detected_streak > 10:
                    search_depth = len(detected)
                if depth > search_depth:
                    last_word_index = 0
                    break
                if percentage >= word_threshold and len(expected[i]) >= 4:
                    last_word_index = j
                    matching_words += 1
                    not_detected_streak = 0
                    search_depth = search_depth_original
                    break
        except IndexError:
            pass
        percentages.append(max(percentages_nested))
    percentage = matching_words / get_number_of_long_words(
        list_name=expected, length=min_word_lenght
    )
    return percentage, percentages


def match_word_percentage(e, d):
    e, d = add_string_parts(e=e, d=d)
    x = create_occurrences(word=e)
    y = create_occurrences(word=d)
    shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    sum_dict = sum(shared_items.values())
    return sum_dict / len(e)


def add_string_parts(e, d):
    e_length = len(e)
    d_length = len(d)
    # print(f"Expected sp {e}")
    # print(f"Detected sp {d}")
    length_diff = abs(e_length - d_length)
    string_diff = "#" * length_diff
    if e_length < d_length:
        e += string_diff
    elif e_length > d_length:
        d += string_diff
    return e, d


def create_list_of_words(text):
    result = []
    for item in text:
        result.append(list(item.split(" ")))
    return result
    # expected_result = list(expected.split(" "))
    # return expected_result


def compare_length(detected, expected):
    text_length = []
    for d, e in zip(detected, expected):
        text_length.append(f"{len(e)},{len(d)}")
    return text_length


def read_files_detected(path):
    file_names_detected = get_file_names(folder_path=path)
    file_paths_detected = create_file_paths(
        file_names=file_names_detected, folder_path=path
    )
    text = []
    for file in file_paths_detected:
        text.append(read_txt_to_str(path=file))
    return text


def read_txt_to_str(path):
    with open(path, "r") as file:
        data = file.read().rstrip()
    return data


def get_file_names(folder_path: str):
    try:
        only_files = [
            f
            for f in sorted_alphanumeric(listdir(folder_path))
            if isfile(join(folder_path, f))
        ]
        only_files.remove(".DS_Store")
    except ValueError:
        print(".Ds_Store not found")
    return only_files


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def create_file_paths(file_names, folder_path):
    file_paths = []
    for item in file_names:
        file_paths.append(f"{folder_path + item}")
    return file_paths


def get_number_of_long_words(list_name, length):
    count = 0
    for item in list_name:
        if len(item) >= length:
            count += 1
    return count


def create_occurrences(word):
    unique = list(set(word))
    occurrences = []
    for item in unique:
        occurrences.append(word.count(item))
    dictionary = dict(zip(unique, occurrences))
    return dictionary


area_bin_edges = [
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]
area_bin_labels = [
    "0.05",
    "0.15",
    "0.25",
    "0.35",
    "0.45",
    "0.55",
    "0.65",
    "0.75",
    "0.85",
    "0.95",
]
