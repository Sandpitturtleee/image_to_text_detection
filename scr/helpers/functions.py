from pathlib import Path

from pdf2image import convert_from_path

from definitions import TRAIN_IMAGES_DIR, TRAIN_PDF_DIR, POPPLER_PATH, RAW_IMAGES_DIR

# pip install pdf2image
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# brew install poppler
# brew --prefix poppler
# pop_path = "/opt/homebrew/opt/poppler/bin"

"""
    A file containing helper functions
"""


def convert_pdf_to_images(pdf_name: str, first_page: int, last_page: int):
    """
        Converts a document in a pdf format to multiple png images corresponding to pages

        Parameters:
        :param pdf_name: Name to the PDF that you want to convert
        :type pdf_name: str
        :param first_page: First page of a PDF file from where you want to start converting.
        :type first_page: int
        :param last_page: Last page of a PDF file to where you want to start converting.
        :type last_page: int
   """
    pages = convert_from_path(
        TRAIN_PDF_DIR + pdf_name, poppler_path=POPPLER_PATH, first_page=first_page, last_page=last_page
    )
    pdf_name = Path(TRAIN_PDF_DIR + pdf_name).stem
    for i in range(len(pages)):
        # Save pages as images in the pdf
        pages[i].save(f"{RAW_IMAGES_DIR}{pdf_name}_page_{str(i)}.png", "PNG")
