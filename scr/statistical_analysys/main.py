from definitions import PAGES_ANALYZE_IMAGES_DIR, PAGES_ANALYZE_LABELS_DIR
from scr.statistical_analysys.functions import (analyze_areas, calculate_areas,
                                                get_bounding_boxes_from_img,
                                                get_bounding_boxes_from_txt,
                                                get_img_sizes)

if __name__ == "__main__":
    print("START")
    img_sizes = get_img_sizes(img_input_path=PAGES_ANALYZE_IMAGES_DIR)

    bounding_boxes_txt = get_bounding_boxes_from_txt(
        txt_input_path=PAGES_ANALYZE_LABELS_DIR,
        img_input_path=PAGES_ANALYZE_IMAGES_DIR,
        img_sizes=img_sizes,
    )
    bounding_boxes_img = get_bounding_boxes_from_img(
        model_name="newspaper_best.pt",
        img_input_path=PAGES_ANALYZE_IMAGES_DIR,
        img_sizes=img_sizes,
    )

    areas_txt = calculate_areas(
        bounding_boxes=bounding_boxes_txt, img_sizes=img_sizes, i=3
    )
    areas_img = calculate_areas(
        bounding_boxes=bounding_boxes_img, img_sizes=img_sizes, i=2
    )

    analyze_areas(areas_txt=areas_txt, areas_img=areas_img)
