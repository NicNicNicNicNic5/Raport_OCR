import os
import cv2
import re
from tqdm import tqdm
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import easyocr

# ───────────────────────────────────────────────────────────
# OCR Model Initialization
# ───────────────────────────────────────────────────────────
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# ───────────────────────────────────────────────────────────
# Folder Paths
# ───────────────────────────────────────────────────────────
DATA_FOLDER = "data"
IMAGES_FOLDER = os.path.join(DATA_FOLDER, "data_images")
PDFS_FOLDER = os.path.join(DATA_FOLDER, "data_pdfs")
GENERATED_IMAGES_FOLDER = os.path.join(IMAGES_FOLDER, "generated")

OUTPUT_FOLDER = "ocr_results"
os.makedirs(GENERATED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ───────────────────────────────────────────────────────────
# Custom Dictionary
# ───────────────────────────────────────────────────────────
CUSTOM_DICTIONARY = [
    "Bahasa Indonesia",
    "Matematika",
    "Bahasa Inggris",
    "Pendidikan Pancasila"
]

# ───────────────────────────────────────────────────────────
# OCR Functions
# ───────────────────────────────────────────────────────────
def perform_ocr(image_path: str) -> str:
    """Run OCR on an image file using PaddleOCR, fallback EasyOCR."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    # PaddleOCR
    results = ocr.ocr(image_path, cls=True)
    text = " ".join([res[1][0] for res in results[0]]) if results and results[0] else ""

    # If Paddle fails → EasyOCR
    if not text or len(text) < 5:
        reader = easyocr.Reader(["id", "en"], gpu=False)
        results = reader.readtext(gray, detail=0)
        text = " ".join(results)

    return text.strip()


def extract_subject_scores(text: str) -> dict:
    """Extract subject scores based on dictionary keywords."""
    extracted_scores = {}
    for subject in CUSTOM_DICTIONARY:
        subject_lower = subject.lower()
        pattern = rf"{subject_lower}\s*([0-9]+)"
        match = re.search(pattern, text.lower())
        if match:
            extracted_scores[subject] = match.group(1)
        else:
            extracted_scores[subject] = "N/A"
    return extracted_scores


def convert_pdf_to_images(pdf_path: str, output_folder: str):
    """Convert PDF into list of image paths."""
    images = convert_from_path(pdf_path, dpi=300)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"{pdf_name}_page_{i}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths


def process_pdf(pdf_path: str):
    """Process a single PDF file and return text + extracted values."""
    image_paths = convert_pdf_to_images(pdf_path, GENERATED_IMAGES_FOLDER)
    all_text = ""
    all_extracted = {}

    for img_path in tqdm(image_paths, desc="Processing PDF pages"):
        text = perform_ocr(img_path)
        extracted_values = extract_subject_scores(text)
        all_text += text + "\n"
        all_extracted.update(extracted_values)

    return all_text.strip(), all_extracted


def process_image(image_path: str):
    """Process a single image file and return text + extracted values."""
    text = perform_ocr(image_path)
    extracted_values = extract_subject_scores(text)
    return text, extracted_values
