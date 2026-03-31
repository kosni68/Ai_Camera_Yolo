# ======================================
# Ultralytics YOLO + EasyOCR
# Automatic Number Plate Recognition
# ======================================

import json
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from capture import FrameGrabber

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
MODEL_PATH = BASE_DIR / "anpr_best.pt"
WINDOW_NAME = "ANPR (Press 'q' to exit)"


def load_runtime_config(config_path):
    config_path = Path(config_path)

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Fichier de configuration introuvable : {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON invalide dans {config_path} : {exc}") from exc

    if not isinstance(raw_config, dict):
        raise RuntimeError(f"Le fichier de configuration doit contenir un objet JSON : {config_path}")

    rtsp_url = str(raw_config.get("rtsp_url", "")).strip()
    if not rtsp_url:
        raise RuntimeError("La cle 'rtsp_url' doit etre renseignee dans config.json.")

    video_display_enabled = raw_config.get("video_display_enabled", False)
    if not isinstance(video_display_enabled, bool):
        raise RuntimeError("La cle 'video_display_enabled' doit etre un booleen.")

    try:
        display_width = int(raw_config.get("display_width", 1280))
        display_height = int(raw_config.get("display_height", 720))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Les cles 'display_width' et 'display_height' doivent etre numeriques.") from exc

    if display_width <= 0 or display_height <= 0:
        raise RuntimeError("Les cles 'display_width' et 'display_height' doivent etre superieures a 0.")

    return {
        "rtsp_url": rtsp_url,
        "video_display_enabled": video_display_enabled,
        "display_width": display_width,
        "display_height": display_height,
    }


def prepare_display_frame(frame, display_width, display_height):
    if frame is None:
        return frame

    frame_height, frame_width = frame.shape[:2]
    scale = min(display_width / frame_width, display_height / frame_height)
    if scale >= 1.0:
        return frame

    resized_width = max(1, int(frame_width * scale))
    resized_height = max(1, int(frame_height * scale))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


class ANPR:
    """Automatic Number Plate Recognition using Ultralytics YOLO and EasyOCR."""

    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(model_path)
        if not model_path.exists():
            raise RuntimeError(f"Modele introuvable : {model_path}")

        self.model = YOLO(str(model_path))
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    def detect_plates(self, im0: np.ndarray):
        """Detects license plates in an image."""
        results = self.model.predict(im0, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else []
        return boxes

    def extract_text(self, im0: np.ndarray, bbox: np.ndarray):
        """Performs OCR on the cropped license plate region."""
        if im0 is None or im0.size == 0:
            return ""

        height, width = im0.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        roi = im0[y1:y2, x1:x2]
        if roi.size == 0:
            return ""

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text = self.reader.readtext(gray, detail=0, paragraph=True)
        return " ".join(part.strip() for part in text if part.strip()).strip()

    def print_detected_plates(self, texts):
        for text in texts:
            print(f"[PLATE] {text}")

    def infer_video(self, source=0, output_path=None, display=True, display_width=1280, display_height=720):
        """Performs real-time ANPR on a video stream."""
        frame_grabber = FrameGrabber(source, queue_size=1)
        writer = None
        last_reported_texts = ()

        try:
            frame_grabber.start()
            frame_grabber.ready.wait(timeout=5)

            if not frame_grabber.ready.is_set():
                raise RuntimeError("La capture video ne repond pas.")
            if frame_grabber.last_error is not None:
                raise frame_grabber.last_error

            frame = frame_grabber.get_latest(timeout=2.0)
            if frame is None:
                raise RuntimeError("Impossible de recuperer une frame initiale depuis la source video.")

            if output_path:
                frame_height, frame_width = frame.shape[:2]
                fps = 30.0
                if frame_grabber.cap is not None:
                    detected_fps = frame_grabber.cap.get(cv2.CAP_PROP_FPS)
                    if detected_fps and detected_fps > 0:
                        fps = detected_fps

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

            if display:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, display_width, display_height)
                print("Starting ANPR live inference. Press 'q' to quit.")
            else:
                print("Starting ANPR live inference. Press Ctrl+C to quit.")

            while True:
                boxes = self.detect_plates(frame)
                detected_texts = []
                annotated_frame = None
                annotator = None

                if display or writer is not None:
                    annotated = frame.copy()
                    annotator = Annotator(annotated, line_width=4)

                for bbox in boxes:
                    text = self.extract_text(frame, bbox)
                    if text:
                        detected_texts.append(text)
                    label = text if text else "plate"
                    if annotator is not None:
                        annotator.box_label(bbox, label=label, color=colors(17, True))

                unique_texts = tuple(dict.fromkeys(detected_texts))
                if unique_texts and unique_texts != last_reported_texts:
                    self.print_detected_plates(unique_texts)
                    last_reported_texts = unique_texts
                elif not unique_texts:
                    last_reported_texts = ()

                if annotator is not None:
                    annotated_frame = annotator.result()

                if display:
                    display_frame = prepare_display_frame(annotated_frame, display_width, display_height)
                    cv2.imshow(WINDOW_NAME, display_frame)
                if writer is not None:
                    writer.write(annotated_frame)

                if display and cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                next_frame = frame_grabber.get_latest(timeout=0.5)
                if next_frame is None:
                    continue
                frame = next_frame
        finally:
            frame_grabber.stop()
            frame_grabber.join(timeout=2)
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()


def main():
    config = load_runtime_config(CONFIG_PATH)
    anpr = ANPR(model_path=MODEL_PATH)
    anpr.infer_video(
        source=config["rtsp_url"],
        output_path=None,
        display=config["video_display_enabled"],
        display_width=config["display_width"],
        display_height=config["display_height"],
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
