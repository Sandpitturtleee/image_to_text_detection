import re
from os import listdir
from os.path import isfile, join
from typing import Any

from definitions import DETECTED_TEXT, EXPECTED_TEXT
from scr.statistical_analysys.plot_functions import create_bar_plot


def analyze_text():
    """
        Analyzes text matching percentage and creates plots for them
    """
    article_percentages, percentages_list = analyze_percentages()
    print(article_percentages)
    for item in percentages_list:
        create_bar_plot(data=item, bin_edges=area_bin_edges, bin_labels=area_bin_labels)


def analyze_percentages() -> tuple[list, list]:
    """
        Read text from txt files and create a list of percentages

        Parameters:
        :return: A tuple of list of percentages
        :rtype: tuple[list,list]
    """
    text_expected = read_files_detected(path=EXPECTED_TEXT)
    text_detected = read_files_detected(path=DETECTED_TEXT)
    converted_detected = create_list_of_words(text=text_detected)
    converted_expected = create_list_of_words(text=text_expected)

    article_percentages, percentages_list = word_matching(
        text_expected=converted_expected,
        text_detected=converted_detected,
        word_threshold=0.75,
        search_depth=10,
    )
    return article_percentages, percentages_list


def word_matching(text_expected: list[list[str]], text_detected: list[list[str]], word_threshold: float,
                  search_depth: int) -> tuple[list, list]:
    """
        Creates tuple of lists of percentages for all txt files

        Parameters:
        :param text_expected: Baseline text to compare to
        :type text_expected: list[list[str]]
        :param text_detected: Detected text
        :type text_detected: list[list[str]]
        :param word_threshold: Percentage threshold where a word is getting classified as recognised correctly
        :type word_threshold: float
        :param search_depth: Amount of steps made in algorythm in search of an appearance of a word in detected text
        :type search_depth: int
        :return: A tuple of list of percentages
        :rtype: tuple[list,list]
    """
    article_percentages = []
    percentages_list = []
    for e, d in zip(text_expected, text_detected):
        article_percentage, percentages = word_matching_article(
            expected=e,
            detected=d,
            word_threshold=word_threshold,
            search_depth=search_depth,
            min_word_length=4,
        )
        article_percentages.append(article_percentage)
        percentages_list.append(percentages)
    return article_percentages, percentages_list


def word_matching_article(
        expected: list[list[str]], detected: list[list[str]], word_threshold: float, search_depth: int,
        min_word_length: int
) -> tuple[float, list[float | Any]]:
    """
        Creates tuple of lists of percentages for txt file in a single article

        Parameters:
        :param expected: Baseline text to compare to
        :type expected: list[list[str]]
        :param detected: Detected text
        :type detected: list[list[str]]
        :param word_threshold: Percentage threshold where a word is getting classified as recognised correctly
        :type word_threshold: float
        :param search_depth: Amount of steps made in algorythm in search of an appearance of a word in detected text
        :type search_depth: int
        :param min_word_length: Minimal length of a word to be accounted in statistics
        :type min_word_length: int
        :return: A tuple of list of percentages
        :rtype: tuple[float, list[float | Any]]
    """
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
        list_name=expected, length=min_word_length
    )
    return percentage, percentages


def match_word_percentage(e: list[str], d: list[str]) -> float:
    """
        Calculates % of words matching parts

        Parameters:
        :param e: Baseline text to compare to
        :type e: list[str]
        :param d: Detected text
        :type d: list[str]
        :return: Percentage as a float ex. 0.5
        :rtype: float
    """
    e, d = add_string_parts(e=e, d=d)
    x = create_occurrences(word=e)
    y = create_occurrences(word=d)
    shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    sum_dict = sum(shared_items.values())
    return sum_dict / len(e)


def add_string_parts(e: list[str], d: list[str]) -> tuple[list, list]:
    """
        Add # to make two string matching in length for further processing

        Parameters:
        :param e: Baseline text to compare to
        :type e: list[str]
        :param d: Detected text
        :type d: list[str]
        :return: New strings with matching length
        :rtype: float
    """
    e_length = len(e)
    d_length = len(d)
    length_diff = abs(e_length - d_length)
    string_diff = "#" * length_diff
    if e_length < d_length:
        e += string_diff
    elif e_length > d_length:
        d += string_diff
    return e, d


def create_list_of_words(text: list[str]) -> list[list[str]]:
    """
        Split text to single words

        Parameters:
        :param text: Text
        :type text: list[str]
        :return: Text split by " " into words
        :rtype: list[list[str]]
    """
    result = []
    for item in text:
        result.append(list(item.split(" ")))
    return result


def read_files_detected(path: str) -> list[str]:
    """
        Read multiple txt files to list

        Parameters:
        :param path: Path to txt file
        :type path: str
        :return: Text read from multiple txt files
        :rtype: list[str]
    """
    file_names_detected = get_file_names(folder_path=path)
    file_paths_detected = create_file_paths(
        file_names=file_names_detected, folder_path=path
    )
    text = []
    for file in file_paths_detected:
        text.append(read_txt_to_str(path=file))
    return text


def read_txt_to_str(path: str) -> str:
    """
        Read single txt file

        Parameters:
        :param path: Path to txt file
        :type path: str
        :return: Text read from txt file
        :rtype: str
    """
    with open(path, "r") as file:
        data = file.read().rstrip()
    return data


def get_file_names(folder_path: str) -> list:
    """
        Read single txt file

        Parameters:
        :param folder_path: Path to folder
        :type folder_path: str
        :return: A list of files
        :rtype: list
    """
    only_files = []
    try:
        only_files = [
            f
            for f in sorted_alphanumeric(listdir(folder_path))
            if isfile(join(folder_path, f))
        ]
        only_files.remove(".DS_Store")
    except ValueError:
        pass
    return only_files


def create_file_paths(file_names: list, folder_path: str)->list:
    """
        Creates path for files

        Parameters:
        :param file_names: Names of files in a folder
        :type file_names: list
        :param folder_path: Path to folder
        :type folder_path: str
        :return: A list of file paths
        :rtype: list
    """
    file_paths = []
    for item in file_names:
        file_paths.append(f"{folder_path + item}")
    return file_paths


def get_number_of_long_words(list_name: list, length: int)->int:
    """
        Creates path for files

        Parameters:
        :param list_name: List with words
        :type list_name: list
        :param length: Length of a word
        :type length: int
        :return: Amount of words of desired length
        :rtype: int
    """
    count = 0
    for item in list_name:
        if len(item) >= length:
            count += 1
    return count


def create_occurrences(word: list) ->dict:
    """
        Creates dictionary of letters in a word with its occurrences

        Parameters:
        :param word: A word
        :type word: list
        :return: Dictionary
        :rtype: dict
    """
    unique = list(set(word))
    occurrences = []
    for item in unique:
        occurrences.append(word.count(item))
    dictionary = dict(zip(unique, occurrences))
    return dictionary


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


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
