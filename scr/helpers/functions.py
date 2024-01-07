from pathlib import Path

from pdf2image import convert_from_path

from definitions import TRAIN_IMAGES_DIR, TRAIN_PDF_DIR, POPPLER_PATH, RAW_IMAGES_DIR


# pip install pdf2image
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# brew install poppler
# brew --prefix poppler
# pop_path = "/opt/homebrew/opt/poppler/bin"


def convert_pdf_to_images(pdf_name: str,first_page: int,last_page: int):
    # Store Pdf with convert_from_path function
    pages = convert_from_path(
        TRAIN_PDF_DIR + pdf_name, poppler_path=POPPLER_PATH, first_page=0,last_page=last_page
    )
    pdf_name = Path(TRAIN_PDF_DIR + pdf_name).stem
    for i in range(len(pages)):
        # Save pages as images in the pdf
        pages[i].save(f"{RAW_IMAGES_DIR}{pdf_name}_page_{str(i)}.png", "PNG")
