from definitions import PAGES_TRAIN_LABELS_DIR, ARTICLES_TRAIN_LABELS_DIR
from scr.statistical_analysys.functions import load_files_and_calculate_areas

if __name__ == "__main__":
    print("START")
    areas = load_files_and_calculate_areas(path=ARTICLES_TRAIN_LABELS_DIR)
    print(areas)