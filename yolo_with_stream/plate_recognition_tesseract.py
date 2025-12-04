import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np
import time
import re
import threading
from queue import Queue, Empty

# --- Fonctions utilitaires ---

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=2, fy=2)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    return thresh


def is_french_plate(text):
    clean = text.replace(" ", "").replace("-", "").upper()
    pattern = r"^([A-Z]{2}\d{3}[A-Z]{2}|\d{4}[A-Z]{2}\d{2}|\d{3}[A-Z]{2}\d{2})$"
    return re.match(pattern, clean) is not None


def format_french_plate(text):
    clean = text.replace(" ", "").replace("-", "").upper()
    if re.match(r"^[A-Z]{2}\d{3}[A-Z]{2}$", clean):
        return f"{clean[:2]}-{clean[2:5]}-{clean[5:]}"
    if re.match(r"^\d{4}[A-Z]{2}\d{2}$", clean):
        return f"{clean[:4]} {clean[4:6]} {clean[6:]}"
    if re.match(r"^\d{3}[A-Z]{2}\d{2}$", clean):
        return f"{clean[:3]} {clean[3:5]} {clean[5:]}"
    return text


def run_tesseract_ocr(image):
    pre = preprocess_for_ocr(image)
    sharp = sharpen_image(pre)
    resized = cv2.resize(sharp, (0, 0), fx=2, fy=2)
    custom_config = r"--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    print("Running OCR with Tesseract")
    text = pytesseract.image_to_string(resized, config=custom_config)
    return text.strip()


def analyze_with_second_model(frame, model):
    print("Running secondary model prediction")
    results = model.predict(frame, imgsz=320)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size > 0:
            return plate_crop
    return np.array([])


def draw_fps_info(frame, fps, min_fps, max_fps):
    fps_text = f"FPS: {fps:.1f} | Min: {min_fps:.1f} | Max: {max_fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (0, 255, 0)
    bg_color = (0, 0, 0)
    (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
    x, y = frame.shape[1] - text_width - 30, text_height + 20
    cv2.rectangle(
        frame,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + 10),
        bg_color,
        -1,
    )
    cv2.putText(frame, fps_text, (x, y), font, font_scale, text_color, font_thickness)
    return frame


def initialize_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir le flux RTSP : {rtsp_url}")
    return cap


class FrameGrabber(threading.Thread):
    """Capture le flux RTSP en tache de fond pour ne pas bloquer l'inference."""

    def __init__(self, rtsp_url, queue_size=3):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.queue = Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.last_error = None
        self.cap = None

    def run(self):
        try:
            self.cap = initialize_rtsp_stream(self.rtsp_url)
            self.ready.set()
        except Exception as exc:
            self.last_error = exc
            self.ready.set()
            return

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
            self.queue.put(frame)

        if self.cap:
            self.cap.release()

    def get_latest(self, timeout=0.5):
        if self.last_error:
            raise self.last_error

        try:
            frame = self.queue.get(timeout=timeout)
        except Empty:
            return None

        while not self.queue.empty():
            try:
                frame = self.queue.get_nowait()
            except Empty:
                break
        return frame

    def stop(self):
        self.stop_event.set()


# --- Main RTSP ---

def main():
    # Mets ici l'URL de ta camera RTSP
    # rtsp_url = "rtsp://user:password@192.168.1.50:554/Streaming/Channels/101"
    rtsp_url = "rtsp://192.168.1.222:8554/mystream"

    print("starting RTSP stream...")
    frame_grabber = FrameGrabber(rtsp_url)
    frame_grabber.start()
    frame_grabber.ready.wait(timeout=5)
    if not frame_grabber.ready.is_set():
        raise RuntimeError("La capture RTSP ne repond pas.")
    if frame_grabber.last_error:
        raise frame_grabber.last_error

    print("Configuring display window...")
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    print("Resizing display window...")
    cv2.resizeWindow("Camera", 1280, 720)

    print("Reading initial frame...")
    frame = frame_grabber.get_latest(timeout=2)
    if frame is None:
        raise RuntimeError("Impossible de recuperer une frame initiale depuis le flux RTSP.")

    print("Loading YOLO models...")

    # Charge le modele principal (yolov8n_ncnn_model ou .pt selon ce que tu veux)
    # model = YOLO("./yolo_with_stream/models/yolov8n.pt")
    model = YOLO("./yolo_with_stream/models/yolov8n_ncnn_model")  # ou "./models/yolov8n.pt"
    license_plate_detector_model = YOLO("./yolo_with_stream/models/license_plate_detector.pt")

    log_file_path = "./yolo_with_stream/data/detected_plates.txt"
    fps_log_file_path = "./yolo_with_stream/data/fps_stats.txt"
    fps_log_interval = 10
    last_fps_log_time = time.time()

    fps_history = []
    last_plate = None

    with open(log_file_path, "a") as log_file, open(fps_log_file_path, "a") as fps_log_file:
        while True:
            start_time = time.time()

            frame = frame_grabber.get_latest()
            if frame is None:
                print("Frame manquante, attente du flux...")
                time.sleep(0.01)
                continue

            # Si besoin : rotation
            # frame = cv2.rotate(frame, cv2.ROTATE_180)

            print("Running YOLO prediction")
            results = model.predict(frame, imgsz=320)[0]
            annotated_frame = results.plot()

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                cv2.imshow("Cropped Plate", plate_crop)

                refined_crop = analyze_with_second_model(plate_crop, license_plate_detector_model)
                if refined_crop.size == 0:
                    continue

                text = run_tesseract_ocr(refined_crop)
                if text:
                    if is_french_plate(text):
                        formatted = format_french_plate(text)
                        if formatted != last_plate:
                            print(f"[VALID] French plate detected: {formatted}")
                            last_plate = formatted
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            log_file.write(f"[{timestamp}] {formatted}\n")
                            log_file.flush()
                    else:
                        print(f"[IGNORED] Non-French plate format: {text}")

                    cv2.putText(
                        refined_crop,
                        text,
                        (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Plate", refined_crop)

            # FPS base sur le temps reel de la boucle
            loop_time = time.time() - start_time
            fps = 1.0 / loop_time if loop_time > 0 else 0.0

            fps_history.append(fps)
            if len(fps_history) > 1000:
                fps_history.pop(0)

            min_fps = min(fps_history) if fps_history else 0.0
            max_fps = max(fps_history) if fps_history else 0.0

            current_time = time.time()
            if current_time - last_fps_log_time >= fps_log_interval:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                fps_log_file.write(f"[{timestamp}] Min FPS: {min_fps:.1f} | Max FPS: {max_fps:.1f}\n")
                fps_log_file.flush()
                fps_history = []
                last_fps_log_time = current_time

            annotated_frame = draw_fps_info(annotated_frame, fps, min_fps, max_fps)
            cv2.imshow("Camera", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    frame_grabber.stop()
    frame_grabber.join(timeout=2)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
