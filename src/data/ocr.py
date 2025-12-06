import os
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import cv2
import pytesseract
from joblib import Memory
import logging

logger = logging.getLogger(__name__)

def init_pdf_processor(cache_dir: str, tesseract_cmd: str, threshold_ocr: int):
    """
    Factory function to create a configured PDFProcessor class with
    a cache directory and OCR settings.
    """
    os.makedirs(cache_dir, exist_ok=True)
    memory = Memory(cache_dir, verbose=0)
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    class PDFProcessor:
        """This class extracts data from pdf files"""

        THRESHOLD_OCR = threshold_ocr

        @staticmethod
        def file_signature(file_path):
            """Generate unique signature for file state"""
            try:
                stat = os.stat(file_path)
                return f"{file_path}-{stat.st_size}-{stat.st_mtime_ns}"
            except Exception as e:
                logger.error(f"Error getting file signature for {file_path}: {e}")
                return None

        @memory.cache
        def extract_text(pdf_path, version_tag):
            """
            Extract text from file. If len(text) < THRESHOLD_OCR, converts pdf
            to image for OCR processing.
            """
            try:
                with fitz.open(pdf_path) as doc:
                    text = ""
                    for page in doc:
                        pix = page.get_pixmap()
                        with Image.frombytes("RGB", [pix.width, pix.height], pix.samples) as img:
                            page_text = page.get_text("text")
                            if len(page_text) < PDFProcessor.THRESHOLD_OCR:
                                page_text += "\n" + pytesseract.image_to_string(img)
                        text += page_text
                    return text
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                return ""

        @memory.cache
        def extract_visual_features(pdf_path, version_tag):
            """Cached visual feature extraction"""
            try:
                with fitz.open(pdf_path) as doc:
                    if len(doc) == 0:
                        return [0] * 4

                    page = doc.load_page(0)
                    pix = page.get_pixmap()
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, 3
                    )

                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    edge_density = np.mean(edges)

                    contours, _ = cv2.findContours(
                        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    contour_count = len(contours)

                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                    white_ratio = np.mean(binary) / 255

                    return [edge_density, contour_count, white_ratio, pix.width * pix.height]
            except Exception as e:
                logger.error(f"Visual feature error {pdf_path}: {e}")
                return [0] * 4

    return PDFProcessor
