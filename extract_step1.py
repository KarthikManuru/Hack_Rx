import os
import fitz  # PyMuPDF
import docx
import email
import extract_msg
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')  # Pre-trained OCR model

INPUT_DIR = "datasets"
OUTPUT_DIR = "output/extracted_text"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_from_pdf(file_path, output_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"[!] Digital PDF extraction failed: {e}. Falling back to OCR.")
        text = extract_pdf_with_ocr(file_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def extract_pdf_with_ocr(file_path):
    text = ""
    images = convert_from_path(file_path)
    for i, image in enumerate(images):
        image_path = f"temp_page_{i}.jpg"
        image.save(image_path)
        result = ocr_model.ocr(image_path, cls=True)
        for line in result:
            for box in line:
                text += box[1][0] + "\n"
        os.remove(image_path)
    return text

def extract_from_docx(file_path, output_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def extract_from_eml(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as f:
        msg = email.message_from_file(f)
    text = f"Subject: {msg.get('subject')}\n"
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
    else:
        text += msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def extract_from_msg(file_path, output_path):
    msg = extract_msg.Message(file_path)
    text = f"Subject: {msg.subject}\nFrom: {msg.sender}\nDate: {msg.date}\n\n{msg.body}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_all_files():
    for filename in os.listdir(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")

        print(f"[+] Processing {filename}")
        if filename.endswith(".pdf"):
            extract_from_pdf(input_path, output_path)
        elif filename.endswith(".docx"):
            extract_from_docx(input_path, output_path)
        elif filename.endswith(".eml"):
            extract_from_eml(input_path, output_path)
        elif filename.endswith(".msg"):
            extract_from_msg(input_path, output_path)
        else:
            print(f"[!] Unsupported file: {filename}")

if __name__ == "__main__":
    process_all_files()
    print("[✔] Extraction Complete. Check the 'output/extracted_text/' folder.")
