""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ # Import libraries                                                       │
  └──────────────────────────────────────────────────────────────────────────┘
 """
import os
import cv2
import numpy as np
import pytesseract
import shutil
from tqdm import tqdm
from PIL import Image
import easyocr
# import torch
# import torchvision
from transformers import pipeline
from fuzzywuzzy import process
from pdf2image import convert_from_path
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR

""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ # Load Models - yang dipake cuma OCR                                     │
  └──────────────────────────────────────────────────────────────────────────┘
 """

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Digitalisasi_UNPAR\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
model = YOLO("yolov8l.pt")
ocr = PaddleOCR(use_angle_cls=True, lang="en")

""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ # Folder Path                                                            │
  └──────────────────────────────────────────────────────────────────────────┘
 """

DATA_FOLDER = "data"
IMAGES_FOLDER = os.path.join(DATA_FOLDER, "data_images")
PDFS_FOLDER = os.path.join(DATA_FOLDER, "data_pdfs")
RAW_IMAGES_FOLDER = os.path.join(IMAGES_FOLDER, "raw")
GENERATED_IMAGES_FOLDER = os.path.join(IMAGES_FOLDER, "generated")
RAW_PDFS_FOLDER = os.path.join(PDFS_FOLDER, "raw")
GENERATED_PDFS_FOLDER = os.path.join(PDFS_FOLDER, "generated")

OUTPUT_FOLDER = "ocr_results"
IMAGES_RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, "images")
PDFS_RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, "pdfs")
TEXT_RESULTS_FOLDER = os.path.join(PDFS_RESULTS_FOLDER, "text_results")
EXTRACTED_VALUES_FOLDER = os.path.join(PDFS_RESULTS_FOLDER, "extracted_values")

# Ensure necessary folders exist
os.makedirs(TEXT_RESULTS_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_VALUES_FOLDER, exist_ok=True)

""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ # Custom Dictionary                                                      │
  │ # TODO: coba buat dictionary pakai name matching strategy                │
  │ # LINK: https://chatgpt.com/share/67d295be-4ed8-8013-b7ee-97bfbd5eeafc   │
  └──────────────────────────────────────────────────────────────────────────┘
 """
# Subject dictionary with correct spellings
CUSTOM_DICTIONARY = ["Bahasa Indonesia", "Matematika", "Bahasa Inggris", "Pendidikan Pancasila"]

""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ # Ambli nilai matpel dari hasil OCR                                      │
  └──────────────────────────────────────────────────────────────────────────┘
 """
# Extract subject scores directly from the OCR result
def extract_subject_scores(text):
    extracted_scores = {}
    text_lines = text.lower().split("\n")  # Convert text to lowercase for matching

    for subject in CUSTOM_DICTIONARY:
        subject_lower = subject.lower()
        pattern = rf"\b{subject_lower}\b\s*([0-9]+)"  # Match exact subject name followed by a number
        match = re.search(pattern, text.lower())

        if match:
            extracted_scores[subject] = match.group(1)  # Extract numerical value

    # Ensure all subjects are present, even if missing from OCR
    for subject in CUSTOM_DICTIONARY:
        if subject not in extracted_scores:
            extracted_scores[subject] = "N/A"

    print("Extracted Values:", extracted_scores)  # Debugging
    return extracted_scores

""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ # Simpan hasil OCR                                                       │
  └──────────────────────────────────────────────────────────────────────────┘
 """
# Save OCR results
def save_results(text, extracted_values, file_name):
    text_file_path = os.path.join(TEXT_RESULTS_FOLDER, f"{file_name}.txt")
    extracted_values_path = os.path.join(EXTRACTED_VALUES_FOLDER, f"{file_name}_values.txt")

    with open(text_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)

    with open(extracted_values_path, "w", encoding="utf-8") as values_file:
        for key, value in extracted_values.items():
            values_file.write(f"{key}: {value}\n")

""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ # Convert PDF ke PNG                                                     │
  │ # TODO: coba extract value langsung dari PDF tanpa convert ke PNG        │
  └──────────────────────────────────────────────────────────────────────────┘
 """
# Convert PDFs to images
def convert_pdf_to_images(pdf_path, output_folder):
    images = convert_from_path(pdf_path, dpi=300)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"{pdf_name}_page_{i}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths

""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ # Access PDF                                                             │
  └──────────────────────────────────────────────────────────────────────────┘
 """
# Process PDFs
def process_pdfs():
    pdf_files = [f for f in os.listdir(RAW_PDFS_FOLDER) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDFs found in the raw PDFs folder.")
        return
    
    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(RAW_PDFS_FOLDER, pdf_file)
        convert_pdf_to_images(pdf_path, GENERATED_IMAGES_FOLDER)
    
    print("PDF processing complete!")

""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ # Fungsi utama OCR, kalo gagal otomatis ganti ke EasyOCR                 │
  └──────────────────────────────────────────────────────────────────────────┘
 """
# OCR Function with PaddleOCR and EasyOCR fallback
def perform_ocr(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    
    # Use PaddleOCR
    results = ocr.ocr(image_path, cls=True)  # Using raw image
    text = " ".join([res[1][0] for res in results[0]]) if results and results[0] else ""
    
    if results and results[0]:
        for res in results[0]:
            word, confidence = res[1]
            print(f"Detected: {word}, Confidence: {confidence}")  # Debugging
            if confidence > 0.5:  # Only keep high-confidence text
                text += word + " "

    text = text.strip()
    print("PaddleOCR Output:", text)  # Debugging output
    
    # If PaddleOCR fails, use EasyOCR
    if not text or len(text) < 5:
        reader = easyocr.Reader(["id", "en"], gpu=True)
        results = reader.readtext(gray, detail=0)
        text = " ".join(results)
        print("EasyOCR Output:", text)  # Debugging output
    
    return text

""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ # Pipeline utama                                                         │
  └──────────────────────────────────────────────────────────────────────────┘
 """
# Main processing pipeline
def main_pipeline():
    process_pdfs()
    image_files = os.listdir(GENERATED_IMAGES_FOLDER)
    
    if not image_files:
        print("No images found in the generated images folder.")
        return

    for img_file in tqdm(image_files):
        img_path = os.path.join(GENERATED_IMAGES_FOLDER, img_file)
        
        if not os.path.isfile(img_path):
            print(f"Skipping: {img_file} (not a valid file)")
            continue
        
        print(f"Processing: {img_file}")
        text = perform_ocr(img_path)
        extracted_values = extract_subject_scores(text)  # Extract values from the OCR result
        save_results(text, extracted_values, img_file.split(".")[0])

    print("Processing complete!")

if __name__ == "__main__":
    main_pipeline()
