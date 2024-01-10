from definitions import PAGES_ANALYZE_IMAGES_DIR, PAGES_ANALYZE_LABELS_DIR
from scr.statistical_analysys.functions import (
    calculate_mismatched_length, calculate_mismatched_zeros,
    calculate_percentages, create_bar_plot, detect_images_and_calculate_areas,
    load_txt_files_and_calculate_areas, sort_nested_list)
from scr.statistical_analysys.variables import (area_bin_edges,
                                                area_bin_labels, percentages)

if __name__ == "__main__":
    print("START")
    # areas_txt = load_txt_files_and_calculate_areas(txt_input_path=PAGES_ANALYZE_LABELS_DIR,img_input_path=PAGES_ANALYZE_IMAGES_DIR)
    #
    # areas_img = detect_images_and_calculate_areas(
    #     model_name="newspaper_best.pt", img_input_path=PAGES_ANALYZE_IMAGES_DIR
    # )
    # areas_txt,areas_img,mismatching_length = calculate_mismatched_length(areas_txt=areas_txt,areas_img=areas_img)
    # areas_txt, areas_img, mismatching_zeros = calculate_mismatched_zeros(areas_txt=areas_txt, areas_img=areas_img)
    # areas_txt = sort_nested_list(nested=areas_txt)
    # areas_img = sort_nested_list(nested=areas_img)
    # percentages = calculate_percentages(areas_txt=areas_txt,areas_img=areas_img)
    print(percentages)
    create_bar_plot(
        data=percentages, bin_edges=area_bin_edges, bin_labels=area_bin_labels
    )
