import pkg_resources
from symspellpy import SymSpell, Verbosity

DICT_PATH = "/Users/sold/Desktop/Python/Projects/image_to_text_detection/scr/text_recognition/pl.txt"


def word_segmentation_symspell(input_term):
    # Set max_dictionary_edit_distance to avoid spelling correction
    sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
    dictionary_path = DICT_PATH

    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    # a sentence without any spaces
    # input_term = "thequickbrownfoxjumpsoverthelazydog"
    result = sym_spell.word_segmentation(input_term)
    print(f"{result.corrected_string}, {result.distance_sum}, {result.log_prob_sum}")


def word_segmentation_symspell1(input_term):
    my_list = input_term.split(" ")
    # Set max_dictionary_edit_distance to avoid spelling correction
    sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
    dictionary_path = DICT_PATH

    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    # a sentence without any spaces
    # input_term = "thequickbrownfoxjumpsoverthelazydog"
    for item in my_list:
        print(item)
        result = sym_spell.word_segmentation(item)
        print(
            f"{result.corrected_string}, {result.distance_sum}, {result.log_prob_sum}"
        )
        print()


def word_correction_symspell(input_term):
    my_list = input_term.split(" ")

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = DICT_PATH
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    # lookup suggestions for single-word input strings
    input_term = "memebers"  # misspelling of "members"
    # max edit distance per lookup
    # (max_edit_distance_lookup <= max_dictionary_edit_distance)

    # display suggestion term, edit distance, and term frequency
    for item in my_list:
        print(item)
        suggestions = sym_spell.lookup(item, Verbosity.CLOSEST, max_edit_distance=2)
        for suggestion in suggestions:
            print(suggestion)
        print()
