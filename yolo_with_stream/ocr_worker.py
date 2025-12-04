import time
import threading
from queue import Queue, Empty
import pytesseract
import cv2

from utils import preprocess_for_ocr, sharpen_image, is_french_plate, format_french_plate


def run_tesseract_ocr(image):
    pre = preprocess_for_ocr(image)
    sharp = sharpen_image(pre)
    resized = cv2.resize(sharp, (0, 0), fx=2, fy=2)
    custom_config = r"--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    print("Running OCR with Tesseract")
    text = pytesseract.image_to_string(resized, config=custom_config)
    return text.strip()


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

    def submit(self, crop):
        if crop is None or crop.size == 0:
            return
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except Empty:
                pass
        self.queue.put(crop)

    def get_latest_plate(self):
        with self.lock:
            return self.latest_text

    def run(self):
        while not self.stop_event.is_set():
            try:
                crop = self.queue.get(timeout=0.2)
            except Empty:
                continue

            text = run_tesseract_ocr(crop)
            if not text:
                continue

            if is_french_plate(text):
                formatted = format_french_plate(text)
                with self.lock:
                    if formatted == self.last_plate:
                        self.latest_text = formatted
                        continue
                    self.last_plate = formatted
                    self.latest_text = formatted
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self.log_file_path, "a") as log_file:
                    log_file.write(f"[{timestamp}] {formatted}\n")
            else:
                with self.lock:
                    self.latest_text = text
                print(f"[IGNORED] Non-French plate format: {text}")

    def stop(self):
        self.stop_event.set()
