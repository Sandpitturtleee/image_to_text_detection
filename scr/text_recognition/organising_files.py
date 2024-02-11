import os
import shutil
from os import listdir
from os.path import isfile, join


from definitions import RESULTS_DIR, NEWSPAPERS_DIR, PAGES_DIR, ARTICLES_DIR, ARTICLES_CROPPED_DIR

def organizing():
    print("Organizing")
    create_newspaper_folders()
    create_pages_folders()
    create_articles_folders()
    #create_results_txt_articles_folder()
    move_files_to_folders()
def move_files_to_folders():
    file_names = get_file_names(folder_path=ARTICLES_CROPPED_DIR)
    for file in file_names:
        n_number = get_newspaper_number(file=file)
        p_number = get_page_number(file=file)
        a_number = get_article_number(file=file)
        folder_path = f"{RESULTS_DIR}{n_number}_newspaper/{p_number}_page/{a_number}_article/"
        file_path = f"{ARTICLES_CROPPED_DIR}{file}"

        shutil.copyfile(file_path, folder_path+file)

def create_newspaper_folders():
    clear_folders()

    file_names = get_file_names(folder_path=NEWSPAPERS_DIR)

    sub_folders_cut = remove_extension(files=file_names, extension="pdf")
    parent_dir = RESULTS_DIR
    for item in sub_folders_cut:
        path = os.path.join(parent_dir, item) + "_newspaper"
        os.mkdir(path)


def create_pages_folders():
    parent_dir = RESULTS_DIR

    file_names = get_file_names(folder_path=PAGES_DIR)
    file_names_cut = remove_extension(files=file_names, extension="png")
    sub_folders_n = get_directory_names(folder_path=parent_dir)

    for sub_folder_n in sub_folders_n:
        folder_path = parent_dir + sub_folder_n
        base_index = sub_folder_n.split("_")[0]
        for item in file_names_cut:
            nested_index = item.split("_")[0]
            nested_name = item.split("_", 1)[1][::-1]
            nested_name = nested_name.split("_", 1)[0]
            if nested_index == base_index:
                path = os.path.join(folder_path, nested_name) + "_page"
                os.mkdir(path)


def create_articles_folders():
    parent_dir = RESULTS_DIR

    file_names = get_file_names(folder_path=ARTICLES_DIR)
    file_names_cut = remove_extension(files=file_names, extension="png")
    sub_folders_n = get_directory_names(folder_path=parent_dir)
    for sub_folder_n in sub_folders_n:
        sub_folder_n_path = os.path.join(RESULTS_DIR, sub_folder_n)
        sub_folders_p = get_directory_names(folder_path=sub_folder_n_path)
        n_index = sub_folder_n.split("_")[0]
        for sub_folder_p in sub_folders_p:
            sub_folder_p_path = os.path.join(sub_folder_n_path, sub_folder_p)
            p_index = sub_folder_p.split("_")[0]
            for item in file_names_cut:
                n_nested_index = item.split("_")[0]
                # 3 rd occurence
                p_nested_index = item.split("_", 2)[-1]
                p_nested_index = p_nested_index.split("_", 1)[0]
                nested_name = item.split("_", 1)[1][::-1]
                nested_name = nested_name.split("_", 1)[0]
                if n_nested_index == n_index and p_nested_index == p_index:
                    path = os.path.join(sub_folder_p_path, nested_name) + "_article"
                    os.mkdir(path)

def create_results_txt_articles_folder():
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
                path = sub_folder_a_path + "/results_txt"
                os.mkdir(path)

def get_file_names(folder_path: str):
    try:
        only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        only_files.remove(".DS_Store")
    except ValueError:
        print(".Ds_Store not found")
    return only_files


def get_directory_names(folder_path: str):
    try:
        sub_folders = [
            name
            for name in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, name))
        ]
        sub_folders.remove(".DS_Store")
    except ValueError:
        print(".Ds_Store not found")
    return sub_folders


def clear_folders():
    """
    Clearing folders from files before every detection and cropping
    """
    folders = [RESULTS_DIR]
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


def remove_extension(files, extension):
    cut_length = 1 + len(extension)
    new_files = []
    for file in files:
        file = file[:-cut_length]
        new_files.append(file)
    return new_files

def create_file_paths(file_names,folder_path):
    file_paths = []
    for item in file_names:
        file_paths.append(f"{folder_path+item}")
    return file_paths

def get_newspaper_number(file):
    n_number = file.split("_")[0]
    return n_number
def get_page_number(file):
    p_number = file.split("_", 2)[-1]
    p_number = p_number.split("_", 1)[0]
    return p_number

def get_article_number(file):
    a_number = file.split("_", 4)[-1]
    a_number = a_number.split("_", 1)[0]
    return a_number

