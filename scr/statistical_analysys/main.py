from definitions import PAGES_ANALYZE_IMAGES_DIR, PAGES_ANALYZE_LABELS_DIR
from scr.statistical_analysys.functions import load_files_and_calculate_areas, detect_images

if __name__ == "__main__":
    print("START")
    areas = load_files_and_calculate_areas(txt_input_path=PAGES_ANALYZE_LABELS_DIR,img_input_path=PAGES_ANALYZE_IMAGES_DIR)
    print(areas)

    #detect_images(model_name="newspaper_best.pt",input_path=PAGES_ANALYZE_IMAGES_DIR)