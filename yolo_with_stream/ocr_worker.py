import time
import threading
from queue import Queue, Empty
import os
import shutil
from collections import Counter, deque
import pytesseract
from pytesseract import TesseractNotFoundError

try:
    import easyocr
except ImportError:
    easyocr = None

from utils import (
    build_ocr_variants,
    format_french_plate,
    is_french_plate,
    normalize_ocr_text,
    prepare_plate_for_ocr,
)


TESSERACT_CONFIGS = (
    ("line", r"--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    ("word", r"--oem 3 --psm 8 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    ("block", r"--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    ("raw", r"--oem 3 --psm 13 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
)

EASYOCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def resolve_tesseract_path():
    for env_var in ("TESSERACT_CMD", "TESSERACT_PATH"):
        candidate = os.environ.get(env_var)
        if candidate and os.path.exists(candidate):
            return candidate

    for command_name in ("tesseract.exe", "tesseract"):
        candidate = shutil.which(command_name)
        if candidate:
            return candidate

    candidates = []
    if os.name == "nt":
        for base_dir in (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            r"C:\Program Files",
            r"C:\Program Files (x86)",
        ):
            if base_dir:
                candidates.append(os.path.join(base_dir, "Tesseract-OCR", "tesseract.exe"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def configure_tesseract():
    tesseract_path = resolve_tesseract_path()
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    return tesseract_path


def create_easyocr_reader():
    if easyocr is None:
        return None

    try:
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:
        print(f"[OCR] EasyOCR indisponible: {exc}")
        return None


def run_easyocr_ocr(image, reader):
    rectified_plate, text_region = prepare_plate_for_ocr(image)
    candidates = []

    for variant_name, variant in (("rectified", rectified_plate), ("text", text_region)):
        results = reader.readtext(
            variant,
            detail=0,
            paragraph=False,
            allowlist=EASYOCR_ALLOWLIST,
        )
        tokens = [normalize_ocr_text(text) for text in results if normalize_ocr_text(text)]
        if not tokens:
            continue

        combined = "".join(tokens)
        candidates.append((variant_name, combined))
        if is_french_plate(combined):
            formatted = format_french_plate(combined)
            print(f"[OCR] Plaque detectee avec EasyOCR: {formatted}")
            return formatted

    if candidates:
        preview = ", ".join(f"{name}={text}" for name, text in candidates[:4])
        print(f"[OCR] Tentatives EasyOCR: {preview}")

    return ""


def run_tesseract_ocr(image):
    best_raw_text = ""
    attempts = []

    for variant_name, variant in build_ocr_variants(image):
        for config_name, config in TESSERACT_CONFIGS:
            text = pytesseract.image_to_string(variant, config=config).strip()
            clean_text = normalize_ocr_text(text)
            if not clean_text:
                continue

            attempts.append(f"{variant_name}/{config_name}={clean_text}")
            if is_french_plate(clean_text):
                formatted = format_french_plate(clean_text)
                print(f"[OCR] Plaque detectee: {formatted}")
                return formatted

            if len(clean_text) > len(best_raw_text):
                best_raw_text = clean_text

    if attempts:
        preview = ", ".join(attempts[:6])
        print(f"[OCR] Tentatives: {preview}")

    return best_raw_text


def run_ocr(image, easyocr_reader=None):
    if easyocr_reader is not None:
        text = run_easyocr_ocr(image, easyocr_reader)
        if text:
            return text

    return run_tesseract_ocr(image)


class PlateOcrWorker(threading.Thread):
    """Traite l'OCR en asynchrone pour ne pas bloquer l'inference principale."""

    def __init__(self, log_file_path, queue_size=5):
        super().__init__(daemon=True)
        self.queue = Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.log_file_path = log_file_path
        self.last_plate = None
        self.latest_text = None
        self.lock = threading.Lock()
        self.recent_plates = deque(maxlen=8)
        self.tesseract_path = configure_tesseract()
        self.easyocr_reader = create_easyocr_reader()
        self.ocr_available = self.easyocr_reader is not None or self.tesseract_path is not None

        if self.easyocr_reader is not None:
            print("[OCR] EasyOCR active")
        if self.tesseract_path:
            print(f"[OCR] Using Tesseract at {self.tesseract_path}")
        if not self.ocr_available:
            print("[OCR] Tesseract introuvable. Definis TESSERACT_CMD ou relance setup_env.ps1.")

    def submit(self, crop):
        if not self.ocr_available or crop is None or crop.size == 0:
            return
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except Empty:
                pass
        self.queue.put(crop.copy())

    def get_latest_plate(self):
        with self.lock:
            return self.latest_text

    def _update_stable_plate(self, formatted):
        self.recent_plates.append(formatted)
        counts = Counter(self.recent_plates)
        stable_plate, stable_count = counts.most_common(1)[0]

        with self.lock:
            self.latest_text = stable_plate
            has_changed = stable_plate != self.last_plate
            if stable_count >= 2 and has_changed:
                self.last_plate = stable_plate
                return stable_plate

        return None

    def run(self):
        if not self.ocr_available:
            return

        while not self.stop_event.is_set():
            try:
                crop = self.queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                text = run_ocr(crop, self.easyocr_reader)
            except TesseractNotFoundError:
                print("[OCR] Tesseract est installe mais n'a pas pu etre lance. Verifie le PATH ou TESSERACT_CMD.")
                if self.easyocr_reader is None:
                    self.ocr_available = False
                    break
            except Exception as exc:
                print(f"[OCR] Erreur pendant la lecture de plaque: {exc}")
                continue

            if not text:
                continue

            if is_french_plate(text):
                formatted = format_french_plate(text)
                stable_plate = self._update_stable_plate(formatted)
                if stable_plate is not None:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(self.log_file_path, "a") as log_file:
                        log_file.write(f"[{timestamp}] {stable_plate}\n")
            else:
                print(f"[IGNORED] Non-French plate format: {text}")

    def stop(self):
        self.stop_event.set()
