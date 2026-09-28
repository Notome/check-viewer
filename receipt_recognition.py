import json
import os
import re
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import cv2
import numpy as np
import ollama
import pymupdf
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image


class PDFExtractor:
    def __init__(self, path: str | None = None):
        self.path = path
        self._ocr = None

    @property
    def ocr(self):
        if self._ocr is None:
            print("Загрузка OCR\n\n")
            self._ocr = PaddleOCR(
                use_textline_orientation=False,
                lang="ru",
                enable_mkldnn=True,
                cpu_threads=4,
            )
        return self._ocr

    def preprocess_image(self, pil_img: Image.Image) -> Image.Image:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        _, w = img.shape[:2]

        if w < 800:
            scale = 800 / w
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        padded = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        return Image.fromarray(padded)

    def extract_direct(self) -> str:
        text_content = []
        with pymupdf.open(self.path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text.strip():
                    text_content.append(page_text)
        return "\n".join(text_content)

    def extract_ocr(self) -> str:
        ext = os.path.splitext(self.path)[1].lower()

        if ext == ".pdf":
            images = convert_from_path(self.path, dpi=150, poppler_path=r"C:\poppler\Library\bin")
        
        elif ext in [".jpg", ".jpeg", ".png"]:
            images = [Image.open(self.path)]
        
        else:
            return ""

        ocr_lines = []

        for img in images:
            processed_img = self.preprocess_image(img)
            temp_img_path = f"_temp_{os.path.basename(self.path)}.png"

            try:
                processed_img.save(temp_img_path, "PNG")
                result = self.ocr.predict(temp_img_path)

                if not result:
                    continue
                
                for res in result:
                    if isinstance(res, dict) and "rec_texts" in res:
                        ocr_lines.extend(res["rec_texts"])
                    elif isinstance(res, list):
                        for line in res:
                            if isinstance(line, list) and len(line) > 1:
                                ocr_lines.append(line[1][0])
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

        return "\n".join(ocr_lines)

    def get_text(self, path: str) -> str:
        if not os.path.exists(path):
            print("Файл не найден")
            return ""

        self.path = path
        _, ext = os.path.splitext(path)
        text = ""

        if ext == ".pdf":
            try:
                text = self.extract_direct()
                if text.strip():
                    print("Текст извлечен из PDF")
            except Exception:
                pass

        if not text.strip():
            print("Начало работы OCR")
            t0 = time.time()
            try:
                text = self.extract_ocr()
                print(f"OCR отработало за {time.time() - t0:.1f} сек")
            except Exception as e:
                print(e)

        return text


def parse_single_receipt(text: str, model_name: str = "qwen2.5:1.5b") -> dict | None:
    prompt_path = ("prompt.txt")

    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    full_prompt = f"{base_prompt}\n\nRECEIPT TEXT TO PROCESS:\n{text}"
    print(" Отправка в qwen")
    
    t0 = time.time()

    client = ollama.Client(timeout=120.0)

    try:
        response = client.generate(
            model=model_name,
            prompt=full_prompt,
            keep_alive="5m",
            options={
                "num_ctx": 2048,
                "num_thread": 4,
                "temperature": 0.0,
            },
        )
        print(f"Ответ получен за {time.time() - t0:.1f} сек")

        raw_output = response["response"]

        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(raw_output)

    except Exception as e:
        return None


if __name__ == "__main__":
    extractor = PDFExtractor()
    paths = ["1.pdf", "2.pdf", "3.png", "4.png"]

    for idx, path in enumerate(paths, 1):
        print(f"\n[{idx}] Обработка файла {path}")

        text = extractor.get_text(path)
        if not text.strip():
            print("Не удалось извлечь текст.")
            continue

        receipt_data = parse_single_receipt(text, model_name="qwen2.5:1.5b")

        if receipt_data:
            output_filename = f"receipt_{idx}.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(receipt_data, f, ensure_ascii=False, indent=2)
            print(f"Сохранен: {output_filename}")