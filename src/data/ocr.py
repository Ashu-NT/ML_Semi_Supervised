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
    cache directory and OCR settings.
    """
    os.makedirs(cache_dir, exist_ok=True)
    memory = Memory(cache_dir, verbose=0)
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    class PDFProcessor:
        """Extracts text and visual features from PDF files."""

        THRESHOLD_OCR = threshold_ocr

        @staticmethod
        def file_signature(file_path):
            """Generate unique signature for file state."""
            try:
                stat = os.stat(file_path)
                return f"{file_path}-{stat.st_size}-{stat.st_mtime_ns}"
            except Exception as e:
                logger.error("Error getting file signature for %s: %s", file_path, e)
                return None

        @staticmethod
        @memory.cache
        def extract_text(pdf_path, version_tag):
            """
            Extract text from PDF.
            If extracted text is below the OCR threshold, OCR is used.
            """
            try:
                text = ""

                with fitz.open(pdf_path) as doc:
                    if len(doc) == 0:
                        logger.warning("PDF has no pages: %s", pdf_path)
                        return ""

                    for page_number, page in enumerate(doc, start=1):
                        page_text = page.get_text("text") or ""

                        if len(page_text.strip()) < PDFProcessor.THRESHOLD_OCR:
                            logger.debug(
                                "OCR used for page %d in file: %s",
                                page_number,
                                pdf_path
                            )

                            # Render page at higher resolution for better OCR
                            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)

                            img = Image.frombytes(
                                "RGB",
                                [pix.width, pix.height],
                                pix.samples
                            )

                            ocr_text = pytesseract.image_to_string(img)
                            page_text += "\n" + ocr_text

                        text += page_text + "\n"

                return text.strip()

            except Exception as e:
                logger.error("Error extracting text from %s: %s", pdf_path, e)
                return ""

        @staticmethod
        @memory.cache
        def extract_visual_features(pdf_path, version_tag):
            """
            Extract basic visual/layout features from the first page of the PDF.
            """
            try:
                with fitz.open(pdf_path) as doc:
                    if len(doc) == 0:
                        logger.warning("PDF has no pages: %s", pdf_path)
                        return [0, 0, 0, 0]

                    page = doc.load_page(0)

                    # Use alpha=False to guarantee RGB-compatible output
                    pix = page.get_pixmap(alpha=False)

                    img_array = np.frombuffer(
                        pix.samples,
                        dtype=np.uint8
                    ).reshape(pix.height, pix.width, 3)

                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                    edges = cv2.Canny(gray, 100, 200)
                    edge_density = float(np.mean(edges))

                    contours, _ = cv2.findContours(
                        edges,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    contour_count = len(contours)

                    _, binary = cv2.threshold(
                        gray,
                        127,
                        255,
                        cv2.THRESH_BINARY
                    )
                    white_ratio = float(np.mean(binary) / 255)

                    page_area = pix.width * pix.height

                    return [
                        edge_density,
                        contour_count,
                        white_ratio,
                        page_area
                    ]

            except Exception as e:
                logger.error("Visual feature error for %s: %s", pdf_path, e)
                return [0, 0, 0, 0]

    return PDFProcessor