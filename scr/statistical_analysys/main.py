from definitions import PAGES_ANALYZE_IMAGES_DIR, PAGES_ANALYZE_LABELS_DIR
from scr.statistical_analysys.functions import (
    load_txt_files_and_calculate_areas,
    detect_images_and_calculate_areas,
)

if __name__ == "__main__":
    print("START")
    areas_txt = load_txt_files_and_calculate_areas(txt_input_path=PAGES_ANALYZE_LABELS_DIR,img_input_path=PAGES_ANALYZE_IMAGES_DIR)

    areas_img = detect_images_and_calculate_areas(
        model_name="newspaper_best.pt", img_input_path=PAGES_ANALYZE_IMAGES_DIR
    )
    for i in range(len(areas_txt)):
        print(areas_txt[i])
        print(areas_img[i])
        print()
