from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.core.config import PROJECT_ROOT


def load_main_detector():
    return YOLO(str(PROJECT_ROOT / "models" / "yolov8n_ncnn_model"), task="detect")


def load_plate_detector(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(f"Secondary plate detector model not found: {model_path}")
    return YOLO(str(model_path), task="detect")


def _sanitize_label(value):
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in value.strip().lower())
    safe = safe.strip("-_")
    return safe or "unknown"


def extract_detection_records(results, x_offset=0, y_offset=0):
    detections = []
    names = results.names if hasattr(results, "names") else {}

    for box in results.boxes:
        cls_id = int(box.cls[0]) if box.cls is not None else -1
        cls_name = names.get(cls_id, str(cls_id)) if hasattr(names, "get") else str(cls_id)
        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append(
            {
                "class_id": cls_id,
                "class_name": str(cls_name),
                "label": _sanitize_label(str(cls_name)),
                "confidence": confidence,
                "x1": x1 + x_offset,
                "y1": y1 + y_offset,
                "x2": x2 + x_offset,
                "y2": y2 + y_offset,
            }
        )

    return detections


def collect_detected_classes(detections, min_confidence=0.0):
    classes = []
    seen = set()

    for detection in detections:
        if detection["confidence"] < min_confidence:
            continue
        normalized = detection["label"]
        if normalized in seen:
            continue
        seen.add(normalized)
        classes.append(normalized)

    return classes


def build_roi_pixels(frame, roi_config):
    if not roi_config["enabled"]:
        return None

    frame_height, frame_width = frame.shape[:2]
    x1 = int(round(frame_width * roi_config["x"]))
    y1 = int(round(frame_height * roi_config["y"]))
    x2 = int(round(frame_width * (roi_config["x"] + roi_config["width"])))
    y2 = int(round(frame_height * (roi_config["y"] + roi_config["height"])))

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))

    return {**roi_config, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def crop_frame_to_roi(frame, roi_pixels):
    if roi_pixels is None:
        return frame
    return frame[roi_pixels["y1"]:roi_pixels["y2"], roi_pixels["x1"]:roi_pixels["x2"]]


def should_run_detector_now(current_perf, last_detector_run_at, detector_interval):
    return last_detector_run_at == 0.0 or (current_perf - last_detector_run_at) >= detector_interval


def box_intersects_roi(detection, roi_pixels):
    if roi_pixels is None:
        return True
    return (
        detection["x1"] < roi_pixels["x2"]
        and detection["x2"] > roi_pixels["x1"]
        and detection["y1"] < roi_pixels["y2"]
        and detection["y2"] > roi_pixels["y1"]
    )


def draw_detected_boxes(frame, detections, roi_pixels=None, copy_frame=True):
    annotated = frame.copy() if copy_frame else frame
    if roi_pixels is not None:
        cv2.rectangle(
            annotated,
            (roi_pixels["x1"], roi_pixels["y1"]),
            (roi_pixels["x2"], roi_pixels["y2"]),
            (255, 200, 0),
            2,
        )

    for detection in detections:
        x1 = max(0, detection["x1"])
        y1 = max(0, detection["y1"])
        x2 = max(x1 + 1, detection["x2"])
        y2 = max(y1 + 1, detection["y2"])
        label = f"{detection['class_name']} {detection['confidence']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return annotated


def save_detection_frame(frame, detected_classes, save_root):
    if frame is None or frame.size == 0 or not detected_classes:
        return []

    timestamp = datetime.now()
    day_folder = timestamp.strftime("%Y-%m-%d")
    file_stamp = timestamp.strftime("%Y%m%d-%H%M%S-%f")[:-3]
    classes_slug = "-".join(detected_classes)
    file_name = f"{file_stamp}__{classes_slug}.jpg"

    saved_paths = []
    for class_name in detected_classes:
        output_path = Path(save_root) / day_folder / class_name / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(output_path), frame):
            saved_paths.append(output_path)
            print(f"[DETECTION] Saved {output_path} | classes={','.join(detected_classes)}")
        else:
            print(f"[DETECTION] Failed to save {output_path}")

    return saved_paths
