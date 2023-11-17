from pdf2image import convert_from_path

from definitions import TRAIN_IMAGES_DIR, TRAIN_PDF_DIR

# pip install pdf2image
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# brew install poppler
# brew --prefix poppler
# pop_path = "/opt/homebrew/opt/poppler/bin"


def convert_pdf_to_images(pdf_name: str):
    pop_path = "/opt/homebrew/opt/poppler/bin"
    # Store Pdf with convert_from_path function
    pages = convert_from_path(
        TRAIN_PDF_DIR + pdf_name, poppler_path=pop_path, last_page=20
    )
    for i in range(len(pages)):
        # Save pages as images in the pdf
        pages[i].save(TRAIN_IMAGES_DIR + "page" + str(i) + ".jpg", "JPEG")
