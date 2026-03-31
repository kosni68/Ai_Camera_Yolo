import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from capture import FrameGrabber
from detection import analyze_with_second_model
from log_utils import (
    active_history_label,
    atomic_write_text,
    append_line,
    current_local_timestamp,
    daily_history_path,
)
from ocr_worker import PlateOcrWorker
from utils import draw_fps_info, VEHICLE_CLASSES

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
REQUIRED_CONFIG_KEYS = (
    "rtsp_url",
    "video_display_enabled",
    "overlay_enabled",
    "ocr_enabled",
    "secondary_plate_detector_enabled",
    "save_detections_enabled",
    "detection_save_min_confidence",
    "detection_save_root",
    "roi_enabled",
    "roi_x",
    "roi_y",
    "roi_width",
    "roi_height",
    "fps_limit",
    "fps_log_interval",
    "fps_summary_interval",
)


def load_runtime_config(config_path):
    config_path = Path(config_path)

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Configuration file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Configuration file is invalid JSON: {config_path} ({exc})") from exc

    if not isinstance(raw_config, dict):
        raise RuntimeError(f"Configuration file must contain a JSON object: {config_path}")

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in raw_config]
    if missing_keys:
        raise RuntimeError(
            f"Configuration file is missing required keys: {', '.join(missing_keys)}"
        )

    rtsp_url = str(raw_config["rtsp_url"]).strip()
    if not rtsp_url:
        raise RuntimeError("Configuration key 'rtsp_url' must not be empty.")

    video_display_enabled = raw_config["video_display_enabled"]
    if not isinstance(video_display_enabled, bool):
        raise RuntimeError("Configuration key 'video_display_enabled' must be a boolean.")

    overlay_enabled = raw_config["overlay_enabled"]
    if not isinstance(overlay_enabled, bool):
        raise RuntimeError("Configuration key 'overlay_enabled' must be a boolean.")

    ocr_enabled = raw_config["ocr_enabled"]
    if not isinstance(ocr_enabled, bool):
        raise RuntimeError("Configuration key 'ocr_enabled' must be a boolean.")

    secondary_plate_detector_enabled = raw_config["secondary_plate_detector_enabled"]
    if not isinstance(secondary_plate_detector_enabled, bool):
        raise RuntimeError("Configuration key 'secondary_plate_detector_enabled' must be a boolean.")

    save_detections_enabled = raw_config["save_detections_enabled"]
    if not isinstance(save_detections_enabled, bool):
        raise RuntimeError("Configuration key 'save_detections_enabled' must be a boolean.")

    roi_enabled = raw_config["roi_enabled"]
    if not isinstance(roi_enabled, bool):
        raise RuntimeError("Configuration key 'roi_enabled' must be a boolean.")

    detection_save_root_value = str(raw_config["detection_save_root"]).strip()
    if not detection_save_root_value:
        raise RuntimeError("Configuration key 'detection_save_root' must not be empty.")

    try:
        detection_save_min_confidence = float(raw_config["detection_save_min_confidence"])
        fps_limit = float(raw_config["fps_limit"])
        fps_log_interval = float(raw_config["fps_log_interval"])
        fps_summary_interval = float(raw_config["fps_summary_interval"])
        detector_fps_limit = float(raw_config.get("detector_fps_limit", min(fps_limit, 10.0)))
        roi_x = float(raw_config["roi_x"])
        roi_y = float(raw_config["roi_y"])
        roi_width = float(raw_config["roi_width"])
        roi_height = float(raw_config["roi_height"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Configuration keys 'detection_save_min_confidence', 'fps_limit', "
            "'fps_log_interval', 'fps_summary_interval', 'detector_fps_limit', "
            "'roi_x', 'roi_y', 'roi_width', and 'roi_height' must be numeric."
        ) from exc

    if fps_limit <= 0:
        raise RuntimeError("Configuration key 'fps_limit' must be greater than 0.")
    if detector_fps_limit <= 0:
        raise RuntimeError("Configuration key 'detector_fps_limit' must be greater than 0.")
    for key, value in (
        ("roi_x", roi_x),
        ("roi_y", roi_y),
        ("roi_width", roi_width),
        ("roi_height", roi_height),
    ):
        if not (0.0 <= value <= 1.0):
            raise RuntimeError(f"Configuration key '{key}' must be between 0.0 and 1.0.")
    if roi_width <= 0.0 or roi_height <= 0.0:
        raise RuntimeError("Configuration keys 'roi_width' and 'roi_height' must be greater than 0.")
    if roi_x + roi_width > 1.0:
        raise RuntimeError("Configuration keys 'roi_x' + 'roi_width' must be <= 1.0.")
    if roi_y + roi_height > 1.0:
        raise RuntimeError("Configuration keys 'roi_y' + 'roi_height' must be <= 1.0.")

    detection_save_root = Path(detection_save_root_value)
    if not detection_save_root.is_absolute():
        detection_save_root = config_path.parent / detection_save_root

    return {
        "rtsp_url": rtsp_url,
        "video_display_enabled": video_display_enabled,
        "overlay_enabled": overlay_enabled,
        "ocr_enabled": ocr_enabled,
        "secondary_plate_detector_enabled": secondary_plate_detector_enabled,
        "save_detections_enabled": save_detections_enabled,
        "detection_save_min_confidence": detection_save_min_confidence,
        "detection_save_root": detection_save_root,
        "detector_fps_limit": min(detector_fps_limit, fps_limit),
        "roi": {
            "enabled": roi_enabled,
            "x": roi_x,
            "y": roi_y,
            "width": roi_width,
            "height": roi_height,
        },
        "fps_limit": fps_limit,
        "fps_log_interval": fps_log_interval,
        "fps_summary_interval": fps_summary_interval,
    }


def display_server_available():
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def adapt_runtime_config_for_environment(config):
    adjusted_config = dict(config)
    adjusted_config["roi"] = dict(config["roi"])

    if adjusted_config["video_display_enabled"] and not display_server_available():
        adjusted_config["video_display_enabled"] = False
        adjusted_config["overlay_enabled"] = False
        print(
            "[CONFIG] No graphical display detected "
            "(DISPLAY/WAYLAND_DISPLAY missing). "
            "Disabling OpenCV windows and overlays for this run."
        )

    return adjusted_config


def apply_fps_limit(loop_started_at, fps_limit):
    target_loop_duration = 1.0 / fps_limit
    elapsed = time.perf_counter() - loop_started_at
    remaining = target_loop_duration - elapsed
    if remaining > 0:
        time.sleep(remaining)
    return time.perf_counter() - loop_started_at


def write_fps_summary(summary_path, session_started_at, current_fps, session_min_fps, session_max_fps):
    lines = [
        f"session_started_at={session_started_at}",
        f"current_fps={current_fps:.1f}",
        f"min_fps={session_min_fps:.1f}",
        f"max_fps={session_max_fps:.1f}",
        f"active_history_file={active_history_label(summary_path)}",
    ]
    atomic_write_text(summary_path, "\n".join(lines) + "\n")


def build_default_ocr_stats(session_started_at, log_file_path):
    return {
        "current_plate": None,
        "consecutive_reads": 0,
        "ocr_jobs_processed": 0,
        "ocr_success_stabilized": 0,
        "ocr_failure_total": 0,
        "ocr_failure_non_french": 0,
        "ocr_failure_unstable": 0,
        "ocr_failure_empty": 0,
        "ocr_success_rate_pct": 0.0,
        "ocr_failure_rate_pct": 0.0,
        "session_started_at": session_started_at,
        "active_history_file": active_history_label(log_file_path),
    }


def write_ocr_summary(log_file_path, stats_info):
    lines = [
        f"current_plate={stats_info['current_plate'] or ''}",
        f"consecutive_reads={stats_info['consecutive_reads']}",
        f"ocr_jobs_processed={stats_info['ocr_jobs_processed']}",
        f"ocr_success_stabilized={stats_info['ocr_success_stabilized']}",
        f"ocr_failure_total={stats_info['ocr_failure_total']}",
        f"ocr_failure_non_french={stats_info['ocr_failure_non_french']}",
        f"ocr_failure_unstable={stats_info['ocr_failure_unstable']}",
        f"ocr_failure_empty={stats_info['ocr_failure_empty']}",
        f"ocr_success_rate_pct={stats_info['ocr_success_rate_pct']:.1f}",
        f"ocr_failure_rate_pct={stats_info['ocr_failure_rate_pct']:.1f}",
        f"session_started_at={stats_info['session_started_at']}",
        f"active_history_file={stats_info['active_history_file']}",
    ]
    atomic_write_text(log_file_path, "\n".join(lines) + "\n")


def load_main_detector():
    return YOLO(str(BASE_DIR / "models" / "yolov8n_ncnn_model"), task="detect")


def load_plate_detector():
    return YOLO(str(BASE_DIR / "models" / "license_plate_detector_ncnn_model"), task="detect")


def load_models(load_secondary_detector=True):
    detector = load_main_detector()
    plate_detector = load_plate_detector() if load_secondary_detector else None
    return detector, plate_detector


def _sanitize_label(value):
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in value.strip().lower())
    safe = safe.strip("-_")
    return safe or "unknown"


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

    return {
        **roi_config,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def crop_frame_to_roi(frame, roi_pixels):
    if roi_pixels is None:
        return frame
    return frame[roi_pixels["y1"]:roi_pixels["y2"], roi_pixels["x1"]:roi_pixels["x2"]]


def box_intersects_roi(detection, roi_pixels):
    if roi_pixels is None:
        return True
    return (
        detection["x1"] < roi_pixels["x2"]
        and detection["x2"] > roi_pixels["x1"]
        and detection["y1"] < roi_pixels["y2"]
        and detection["y2"] > roi_pixels["y1"]
    )


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


def build_display_frame(
    frame,
    detections,
    roi_pixels,
    latest_plate_info,
    stats_info,
    fps,
    min_fps,
    max_fps,
):
    annotated_frame = draw_detected_boxes(frame, detections, roi_pixels=roi_pixels)

    if latest_plate_info:
        latest_plate = latest_plate_info["plate"]
        consecutive_count = latest_plate_info["consecutive_count"]
        cv2.putText(
            annotated_frame,
            f"Derniere plaque: {latest_plate} (x{consecutive_count})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
        )

    cv2.putText(
        annotated_frame,
        f"OCR OK: {stats_info['ocr_success_rate_pct']:.1f}% | Echec: {stats_info['ocr_failure_rate_pct']:.1f}% | Jobs: {stats_info['ocr_jobs_processed']}",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 255, 160),
        2,
    )

    return draw_fps_info(annotated_frame, fps, min_fps, max_fps)


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


def main():
    config = adapt_runtime_config_for_environment(load_runtime_config(CONFIG_PATH))

    log_file_path = BASE_DIR / "data" / "detected_plates.txt"
    fps_log_file_path = BASE_DIR / "data" / "fps_stats.txt"
    fps_log_interval = config["fps_log_interval"]
    fps_summary_interval = config["fps_summary_interval"]
    session_started_at = current_local_timestamp()
    rtsp_url = config["rtsp_url"]
    video_display_enabled = config["video_display_enabled"]
    overlay_enabled = config["overlay_enabled"]
    ocr_enabled = config["ocr_enabled"]
    secondary_plate_detector_enabled = config["secondary_plate_detector_enabled"]
    save_detections_enabled = config["save_detections_enabled"]
    detection_save_min_confidence = config["detection_save_min_confidence"]
    detection_save_root = config["detection_save_root"]
    detector_fps_limit = config["detector_fps_limit"]
    roi_config = config["roi"]
    fps_limit = config["fps_limit"]
    default_ocr_stats = build_default_ocr_stats(session_started_at, log_file_path)

    print(f"[CONFIG] Loaded {CONFIG_PATH}")
    print(f"[CONFIG] RTSP URL: {rtsp_url}")
    print(f"[CONFIG] Video display: {'ON' if video_display_enabled else 'OFF'}")
    print(f"[CONFIG] Overlay: {'ON' if overlay_enabled else 'OFF'}")
    print(f"[CONFIG] OCR: {'ON' if ocr_enabled else 'OFF'}")
    print(
        f"[CONFIG] Secondary plate detector: "
        f"{'ON' if secondary_plate_detector_enabled else 'OFF'}"
    )
    print(f"[CONFIG] Save detections: {'ON' if save_detections_enabled else 'OFF'}")
    print(f"[CONFIG] FPS limit: {fps_limit:.1f}")
    print(f"[CONFIG] Detector FPS limit: {detector_fps_limit:.1f}")
    if roi_config["enabled"]:
        print(
            f"[CONFIG] ROI: ON x={roi_config['x']:.2f} y={roi_config['y']:.2f} "
            f"w={roi_config['width']:.2f} h={roi_config['height']:.2f}"
        )
    else:
        print("[CONFIG] ROI: OFF")

    print("starting RTSP stream...")
    frame_grabber = FrameGrabber(rtsp_url, queue_size=1)
    ocr_worker = None
    model = None
    license_plate_detector_model = None

    try:
        frame_grabber.start()
        frame_grabber.ready.wait(timeout=5)
        if not frame_grabber.ready.is_set():
            raise RuntimeError("La capture RTSP ne repond pas.")
        if frame_grabber.last_error:
            raise frame_grabber.last_error

        if video_display_enabled:
            print("Configuring display window...")
            cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera", 1280, 720)

        print("Reading initial frame...")
        frame = frame_grabber.get_latest(timeout=2)
        if frame is None:
            raise RuntimeError("Impossible de recuperer une frame initiale depuis le flux RTSP.")

        print("Loading YOLO models...")
        model = load_main_detector()
        if ocr_enabled and secondary_plate_detector_enabled:
            license_plate_detector_model = load_plate_detector()
        if save_detections_enabled:
            print(f"[DETECTION] Save root: {detection_save_root}")
            print(f"[DETECTION] Min confidence: {detection_save_min_confidence:.2f}")

        if ocr_enabled:
            ocr_worker = PlateOcrWorker(log_file_path)
            ocr_worker.start()
        else:
            write_ocr_summary(log_file_path, default_ocr_stats)

        fps_history = []
        session_min_fps = 0.0
        session_max_fps = 0.0
        detector_interval = 1.0 / detector_fps_limit
        last_detector_run_at = 0.0
        last_detections = []
        last_fps_log_time = time.time()
        last_fps_summary_time = 0.0
        write_fps_summary(fps_log_file_path, session_started_at, 0.0, session_min_fps, session_max_fps)

        while True:
            loop_started_at = time.perf_counter()

            frame = frame_grabber.get_latest()
            if frame is None:
                print("Frame manquante, attente du flux...")
                time.sleep(0.01)
                continue

            fresh_detections = []
            roi_pixels = build_roi_pixels(frame, roi_config)
            current_perf = time.perf_counter()
            if last_detector_run_at == 0.0 or (current_perf - last_detector_run_at) >= detector_interval:
                detection_frame = crop_frame_to_roi(frame, roi_pixels)
                results = model.predict(detection_frame, imgsz=320, verbose=False)[0]
                x_offset = roi_pixels["x1"] if roi_pixels is not None else 0
                y_offset = roi_pixels["y1"] if roi_pixels is not None else 0
                fresh_detections = extract_detection_records(results, x_offset=x_offset, y_offset=y_offset)
                if roi_pixels is not None:
                    fresh_detections = [
                        detection for detection in fresh_detections if box_intersects_roi(detection, roi_pixels)
                    ]
                last_detections = fresh_detections
                last_detector_run_at = time.perf_counter()

                if save_detections_enabled and fresh_detections:
                    detected_classes = collect_detected_classes(fresh_detections, detection_save_min_confidence)
                    if detected_classes:
                        save_detection_frame(
                            draw_detected_boxes(frame, fresh_detections, roi_pixels=roi_pixels),
                            detected_classes,
                            detection_save_root,
                        )

            submitted_this_frame = False

            if ocr_worker is not None:
                for detection in fresh_detections:
                    if submitted_this_frame or not ocr_worker.ready_for_new_job():
                        break

                    if detection["label"] not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    if video_display_enabled:
                        cv2.imshow("Cropped Plate", plate_crop)

                    refined_crop = plate_crop
                    if license_plate_detector_model is not None:
                        refined_crop = analyze_with_second_model(plate_crop, license_plate_detector_model)
                        if refined_crop.size == 0:
                            continue

                    if ocr_worker.submit(refined_crop):
                        submitted_this_frame = True
                        if video_display_enabled:
                            cv2.imshow("Plate", refined_crop)

            total_loop_time = apply_fps_limit(loop_started_at, fps_limit)
            fps = 1.0 / total_loop_time if total_loop_time > 0 else 0.0

            fps_history.append(fps)
            if len(fps_history) > 1000:
                fps_history.pop(0)

            min_fps = min(fps_history) if fps_history else 0.0
            max_fps = max(fps_history) if fps_history else 0.0
            session_min_fps = fps if session_min_fps == 0.0 else min(session_min_fps, fps)
            session_max_fps = max(session_max_fps, fps)

            current_time = time.time()
            if current_time - last_fps_log_time >= fps_log_interval:
                append_line(
                    daily_history_path(fps_log_file_path),
                    f"[{current_local_timestamp()}] Min FPS: {min_fps:.1f} | Max FPS: {max_fps:.1f}",
                )
                fps_history = []
                last_fps_log_time = current_time

            if current_time - last_fps_summary_time >= fps_summary_interval:
                write_fps_summary(
                    fps_log_file_path,
                    session_started_at,
                    fps,
                    session_min_fps,
                    session_max_fps,
                )
                last_fps_summary_time = current_time

            if video_display_enabled:
                latest_plate_info = (
                    ocr_worker.get_latest_plate_info()
                    if ocr_worker is not None
                    else None
                )
                stats_info = (
                    ocr_worker.get_stats_info()
                    if ocr_worker is not None
                    else default_ocr_stats
                )
                camera_frame = frame
                if overlay_enabled:
                    camera_frame = build_display_frame(
                        frame,
                        last_detections,
                        roi_pixels,
                        latest_plate_info,
                        stats_info,
                        fps,
                        min_fps,
                        max_fps,
                    )

                cv2.imshow("Camera", camera_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        write_fps_summary(
            fps_log_file_path,
            session_started_at,
            fps if "fps" in locals() else 0.0,
            session_min_fps,
            session_max_fps,
        )
        print(f"[FPS] Active history file: {active_history_label(fps_log_file_path)}")
    finally:
        if ocr_worker is not None:
            ocr_worker.stop()
            ocr_worker.join(timeout=2)
        frame_grabber.stop()
        frame_grabber.join(timeout=2)
        if video_display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
