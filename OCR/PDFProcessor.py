import concurrent.futures
import io
import os
from typing import List

import fitz
from PIL import Image
import pytesseract


class PDFProcessor:

    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: str) -> List[bytes]:
        """
        Convert PDF pages to images using 4 workers.

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image conversion

        Returns:
            List of image bytes for each page
        """
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()
            if num_pages == 0:
                return []

            images: List[bytes | None] = []

            def _render_page(page_num: int) -> bytes | None:
                try:
                    d = fitz.open(pdf_path)
                    page = d.load_page(page_num)
                    mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    d.close()
                    return img_data
                except Exception as e:
                    print(f"Error converting page {page_num + 1} to image: {e}")
                    return None

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, img in enumerate(executor.map(_render_page, range(num_pages))):
                    if (i + 1) % 50 == 0:
                        print(f"Converted {i + 1}/{num_pages} pages to images...")
                    images.append(img)

            return [img for img in images if img is not None]

        except Exception as e:
            print(f"Error converting PDF to images: {e}")
            return []

    def ocr_pdf(self, pdf_path: str) -> List[str]:
        """
        Perform OCR on a PDF file and return extracted text for each page.

        Args:
            pdf_path: Path to the PDF file
        Returns:
            List of extracted text for each page
        """
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return []

        images = self.pdf_to_images(pdf_path)
        if not images:
            return []
        print(f"Starting OCR on {len(images)} pages...")

        def _ocr_image(img_data: bytes) -> str | None:
            try:
                with Image.open(io.BytesIO(img_data)) as img:
                    text = pytesseract.image_to_string(img)
                return text if text and text.strip() != "" else None
            except Exception as e:
                print(f"OCR error on one page: {e}")
                return None

        result: List[str | None] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i, text in enumerate(executor.map(_ocr_image, images)):
                if (i + 1) % 50 == 0:
                    print(f"OCR processed {i + 1}/{len(images)} pages...")
                result.append(text)

        return result
