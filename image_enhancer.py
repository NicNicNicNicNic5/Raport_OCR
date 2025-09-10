import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import Augmentor
from tqdm import tqdm

# Define paths
DATA_FOLDER = "data"
PDFS_FOLDER = os.path.join(DATA_FOLDER, "data_pdfs")
PROCESSED_IMAGES_FOLDER = os.path.join(PDFS_FOLDER, "processed_image")
GENERATED_IMAGES_FOLDER = os.path.join(PDFS_FOLDER, "generated")

# Ensure necessary folders exist
os.makedirs(PROCESSED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(GENERATED_IMAGES_FOLDER, exist_ok=True)

# Convert PDFs to images and enhance them
def convert_and_enhance_pdfs():
    pdf_files = [f for f in os.listdir(PDFS_FOLDER) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDFs found in the folder.")
        return
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(PDFS_FOLDER, pdf_file)
        images = convert_from_path(pdf_path, dpi=300)
        pdf_name = os.path.splitext(pdf_file)[0]
        
        for i, image in enumerate(images):
            image_path = os.path.join(PROCESSED_IMAGES_FOLDER, f"{pdf_name}_page_{i}.png")
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            enhanced_image = enhance_image(image)
            cv2.imwrite(image_path, enhanced_image)

# Enhance image quality
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return enhanced

# Augment images using Augmentor
def augment_images():
    p = Augmentor.Pipeline(PROCESSED_IMAGES_FOLDER, GENERATED_IMAGES_FOLDER)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.flip_left_right(probability=0.5)
    p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)
    p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.3)
    p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=5)
    p.sample(50)  # Generate 50 augmented images

if __name__ == "__main__":
    convert_and_enhance_pdfs()
    augment_images()
    print("Image enhancement and augmentation complete!")
