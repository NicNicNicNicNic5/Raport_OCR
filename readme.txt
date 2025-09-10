# OCR Pipeline README

## Overview
This OCR pipeline processes images containing school subjects and numerical scores, extracts text using Tesseract OCR, and enhances extraction using NLP techniques. The pipeline follows these steps:

1. **Image Augmentation**: Generates multiple variations of an image (grayscale, shift, rotation).
2. **Preprocessing**: Enhances images (DPI adjustment, skew correction, text detection).
3. **OCR Processing**: Uses Tesseract for text recognition.
4. **Entity Extraction**: Utilizes NLP (NER and fuzzy matching) to extract relevant entities.
5. **Results Storage**: Outputs extracted text and structured values into separate folders.

## Installation
Ensure you have Python 3.8+ installed. Install dependencies using (I'm using Python 3.11.9):
```
1. set up virtual environment: run this command in terminal "python -m venv .venv"
2. access the virtual environtment: run this command in terminal ".venv\Scripts\activate"
3. install the required libraries from requirements.txt: run this command in terminal "pip install -r requirements.txt"

If your encounter error on:
no. 2: microsoft store dependency or something:
run this command in terminal "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"
run this command in terminal ".venv\Scripts\Activate.ps1"

no. 3: library installation failed:
manually import the library in the terminal, check the document in web
```

## Usage
1. Place your image in the working directory.
2. Update `test_image` in `main_pipeline(image_path)` with the image filename.
3. Run the script in a Python environment:
   ```
   python ocr_pipeline.py
   ```
4. Results will be saved in `ocr_results/` with:
   - `text_results/` containing raw OCR text.
   - `extracted_values/` containing formatted subject-score mappings.

## Dependencies
See `requirements.txt` for required packages.

## Notes
- Ensure Tesseract OCR is installed and configured.
- The model uses `dbmdz/bert-large-cased-finetuned-conll03-english` for entity extraction.
- Customize the school subjects dictionary in the script for better accuracy.

## License
This project is for educational purposes. Modify and use at your own discretion.

## Folder Structure
📂 OCR
|   📂 data
|   │── 📂 data_images
|   │   │── 📂 raw
|   │   │── 📂 generated
|   │
|   │── 📂 data_pdfs
|   │   │── 📂 raw
|   │   │── 📂 generated
|   │   │── 📂 processed_image
|   │
|   📂 ocr_results
|   │── 📂 images
|   │   │── 📂 text_results
|   │   │── 📂 extracted_values
|   │
|   │── 📂 pdfs
|   │   │── 📂 text_results
|   │   │── 📂 extracted_values

### Install Poppler
**Windows:**
1. Download Poppler from:
   - [https://github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases)
2. Extract the ZIP file (e.g., `poppler-23.11.0`).
3. Copy the extracted folder path (e.g., `C:\poppler-23.11.0\bin`).
4. Add the **bin** folder to your system PATH:
   - Open **System Properties** → **Environment Variables**.
   - Find `Path` under **System Variables**, click **Edit**.
   - Click **New**, paste `C:\poppler-23.11.0\bin`, then click **OK**.
5. Restart your terminal.

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install poppler-utils
```

**MacOS:**
```bash
brew install poppler
```

### Install Tesseract OCR

**Windows:**
1. Download Tesseract from:
   - [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install it and note the installation path (e.g., `C:\Program Files\Tesseract-OCR`).
3. Add this path to your system PATH.

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**MacOS:**
```bash
brew install tesseract
```

### Verify Installation

Run the following to verify:
```bash
tesseract -v
pdfinfo -v
```

## Running the OCR Pipeline

Ensure your images and PDFs are placed in the correct directories:
- Place raw images in: `data/data_images/raw/`
- Place raw PDFs in: `data/data_pdfs/raw/`

Run the script:
```bash
python ocr_pipeline.py
```

Results will be stored in:
- Text Results: `ocr_results/images/text_results/` and `ocr_results/pdfs/text_results/`
- Extracted Values: `ocr_results/images/extracted_values/` and `ocr_results/pdfs/extracted_values/`

### Troubleshooting
- If you see **"Poppler not found"**, ensure you installed it and added it to `PATH`.
- If you see **"Tesseract not found"**, ensure it is installed and in `PATH`.

Enjoy your OCR pipeline!