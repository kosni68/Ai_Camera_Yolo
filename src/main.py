import json
import time

import cv2

from src.core.capture import FrameGrabber
from src.core.config import (
    CONFIG_PATH,
    PROJECT_ROOT,
    adapt_runtime_config_for_environment,
    load_runtime_config,
)
from src.core.motion import MotionDetector
from src.detection import VEHICLE_CLASSES
from src.detection.plate import analyze_with_second_model
from src.detection.yolo import (
    box_intersects_roi,
    build_roi_pixels,
    collect_detected_classes,
    crop_frame_to_roi,
    draw_detected_boxes,
    extract_detection_records,
    load_main_detector,
    load_plate_detector,
    save_detection_frame,
    should_run_detector_now,
)
from src.ocr.worker import PlateOcrWorker
from src.utils.drawing import draw_fps_info
from src.utils.logging import (
    active_history_label,
    append_line,
    atomic_write_text,
    current_local_timestamp,
    daily_history_path,
)
from src.utils.mqtt_client import ShellyMqttTrigger

LOG_FILE_PATH = PROJECT_ROOT / "data" / "detected_plates.txt"
FPS_LOG_FILE_PATH = PROJECT_ROOT / "data" / "fps_stats.txt"


def load_registered_plates(plates_path):
    """Charge la liste des plaques autorisees depuis le fichier JSON.

    La comparaison se fait sur la forme normalisee (sans tirets, majuscules).
    """
    try:
        raw = json.loads(plates_path.read_text(encoding="utf-8"))
        plates = raw.get("plates", [])
        normalized = {p.replace("-", "").upper() for p in plates if isinstance(p, str)}
        print(f"[PLATES] {len(normalized)} plaque(s) pre-enregistree(s) chargee(s) depuis {plates_path}")
        return normalized
    except FileNotFoundError:
        print(f"[PLATES] Fichier introuvable: {plates_path}. Aucune plaque pre-enregistree.")
        return set()
    except Exception as exc:
        print(f"[PLATES] Erreur de chargement ({exc}). Aucune plaque pre-enregistree.")
        return set()


def is_registered_plate(plate, registered_plates):
    return plate.replace("-", "").upper() in registered_plates


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


def build_display_frame(frame, detections, roi_pixels, latest_plate_info, stats_info, fps, min_fps, max_fps):
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


def main():
    config = adapt_runtime_config_for_environment(load_runtime_config())

    fps_log_interval = config["fps_log_interval"]
    fps_summary_interval = config["fps_summary_interval"]
    session_started_at = current_local_timestamp()
    rtsp_url = config["rtsp_url"]
    video_display_enabled = config["video_display_enabled"]
    overlay_enabled = config["overlay_enabled"]
    ocr_enabled = config["ocr_enabled"]
    ocr_fast_mode_enabled = config["ocr_fast_mode_enabled"]
    ocr_submit_interval_sec = config["ocr_submit_interval_sec"]
    ocr_same_crop_retry_sec = config["ocr_same_crop_retry_sec"]
    secondary_plate_detector_enabled = config["secondary_plate_detector_enabled"]
    secondary_plate_detector_model_path = config["secondary_plate_detector_model_path"]
    save_detections_enabled = config["save_detections_enabled"]
    detection_save_min_confidence = config["detection_save_min_confidence"]
    detection_save_root = config["detection_save_root"]
    detector_fps_limit = config["detector_fps_limit"]
    roi_config = config["roi"]
    motion_config = config["motion"]
    fps_limit = config["fps_limit"]
    default_ocr_stats = build_default_ocr_stats(session_started_at, LOG_FILE_PATH)

    mqtt_config = config["mqtt"]
    registered_plates_path = config["registered_plates_path"]

    print(f"[CONFIG] Loaded {CONFIG_PATH}")
    print(f"[CONFIG] RTSP URL: {rtsp_url}")
    print(f"[CONFIG] Video display: {'ON' if video_display_enabled else 'OFF'}")
    print(f"[CONFIG] Overlay: {'ON' if overlay_enabled else 'OFF'}")
    print(f"[CONFIG] OCR: {'ON' if ocr_enabled else 'OFF'}")
    if ocr_enabled:
        print(f"[CONFIG] OCR fast mode: {'ON' if ocr_fast_mode_enabled else 'OFF'}")
        print(
            f"[CONFIG] OCR cadence: submit={ocr_submit_interval_sec:.2f}s "
            f"same-crop-retry={ocr_same_crop_retry_sec:.2f}s"
        )
    print(f"[CONFIG] Secondary plate detector: {'ON' if secondary_plate_detector_enabled else 'OFF'}")
    print(f"[CONFIG] Secondary plate detector model: {secondary_plate_detector_model_path}")
    print(f"[CONFIG] Save detections: {'ON' if save_detections_enabled else 'OFF'}")
    print(f"[CONFIG] FPS limit: {fps_limit:.1f}")
    print(f"[CONFIG] Detector FPS limit: {detector_fps_limit:.1f}")
    if motion_config["enabled"]:
        print(
            "[CONFIG] Motion gate: ON "
            f"width={motion_config['resize_width']} "
            f"diff={motion_config['diff_threshold']} "
            f"area={motion_config['min_area_ratio']:.3f} "
            f"keepalive={motion_config['keepalive_sec']:.2f}s "
            f"force-run={motion_config['force_detector_interval_sec']:.2f}s"
        )
    else:
        print("[CONFIG] Motion gate: OFF")
    if roi_config["enabled"]:
        print(
            f"[CONFIG] ROI: ON x={roi_config['x']:.2f} y={roi_config['y']:.2f} "
            f"w={roi_config['width']:.2f} h={roi_config['height']:.2f}"
        )
    else:
        print("[CONFIG] ROI: OFF")

    registered_plates = load_registered_plates(registered_plates_path)

    mqtt_trigger = None
    if mqtt_config["enabled"]:
        try:
            mqtt_trigger = ShellyMqttTrigger(
                broker_host=mqtt_config["broker_host"],
                broker_port=mqtt_config["broker_port"],
                username=mqtt_config["username"],
                password=mqtt_config["password"],
                shelly_device_id=mqtt_config["shelly_device_id"],
                pulse_duration_sec=mqtt_config["pulse_duration_sec"],
            )
            print(
                f"[CONFIG] MQTT: ON | broker={mqtt_config['broker_host']}:{mqtt_config['broker_port']} "
                f"| Shelly={mqtt_config['shelly_device_id']} | pulse={mqtt_config['pulse_duration_sec']}s"
            )
        except Exception as exc:
            print(f"[CONFIG] MQTT desactive suite a une erreur: {exc}")
            mqtt_trigger = None
    else:
        print("[CONFIG] MQTT: OFF")

    def on_stable_plate(plate, consecutive_count):
        if consecutive_count != 1:
            return
        if is_registered_plate(plate, registered_plates):
            print(f"[ACCESS] Plaque autorisee: {plate} -> signal Shelly")
            if mqtt_trigger is not None:
                mqtt_trigger.trigger(plate)
        else:
            print(f"[ACCESS] Plaque non enregistree: {plate}")

    print("starting RTSP stream...")
    frame_grabber = FrameGrabber(rtsp_url, queue_size=1)
    ocr_worker = None
    model = None
    license_plate_detector_model = None
    motion_detector = None

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
            license_plate_detector_model = load_plate_detector(secondary_plate_detector_model_path)
        if motion_config["enabled"]:
            motion_detector = MotionDetector(
                resize_width=motion_config["resize_width"],
                diff_threshold=motion_config["diff_threshold"],
                min_area_ratio=motion_config["min_area_ratio"],
                keepalive_sec=motion_config["keepalive_sec"],
            )
        if save_detections_enabled:
            print(f"[DETECTION] Save root: {detection_save_root}")
            print(f"[DETECTION] Min confidence: {detection_save_min_confidence:.2f}")

        if ocr_enabled:
            ocr_worker = PlateOcrWorker(
                LOG_FILE_PATH,
                fast_mode_enabled=ocr_fast_mode_enabled,
                submit_interval_sec=ocr_submit_interval_sec,
                same_crop_retry_sec=ocr_same_crop_retry_sec,
                on_stable_plate=on_stable_plate,
            )
            ocr_worker.start()
        else:
            write_ocr_summary(LOG_FILE_PATH, default_ocr_stats)

        fps_history = []
        session_min_fps = 0.0
        session_max_fps = 0.0
        detector_interval = 1.0 / detector_fps_limit
        last_detector_run_at = 0.0
        last_detections = []
        last_fps_log_time = time.time()
        last_fps_summary_time = 0.0
        write_fps_summary(FPS_LOG_FILE_PATH, session_started_at, 0.0, session_min_fps, session_max_fps)

        while True:
            loop_started_at = time.perf_counter()

            frame = frame_grabber.get_latest()
            if frame is None:
                print("Frame manquante, attente du flux...")
                time.sleep(0.01)
                continue

            fresh_detections = []
            roi_pixels = build_roi_pixels(frame, roi_config)
            detection_frame = crop_frame_to_roi(frame, roi_pixels)
            current_perf = time.perf_counter()
            detector_due = should_run_detector_now(current_perf, last_detector_run_at, detector_interval)
            allow_detector_run = last_detector_run_at == 0.0
            if motion_detector is not None:
                motion_state = motion_detector.update(detection_frame, current_perf)
                allow_detector_run = allow_detector_run or motion_state["recent_motion"]
                force_detector_run = (
                    last_detector_run_at != 0.0
                    and (current_perf - last_detector_run_at) >= motion_config["force_detector_interval_sec"]
                )
                allow_detector_run = allow_detector_run or force_detector_run

            if detector_due and allow_detector_run:
                results = model.predict(detection_frame, imgsz=320, verbose=False)[0]
                x_offset = roi_pixels["x1"] if roi_pixels is not None else 0
                y_offset = roi_pixels["y1"] if roi_pixels is not None else 0
                fresh_detections = extract_detection_records(results, x_offset=x_offset, y_offset=y_offset)
                if roi_pixels is not None:
                    fresh_detections = [
                        det for det in fresh_detections if box_intersects_roi(det, roi_pixels)
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
                    daily_history_path(FPS_LOG_FILE_PATH),
                    f"[{current_local_timestamp()}] Min FPS: {min_fps:.1f} | Max FPS: {max_fps:.1f}",
                )
                fps_history = []
                last_fps_log_time = current_time

            if current_time - last_fps_summary_time >= fps_summary_interval:
                write_fps_summary(FPS_LOG_FILE_PATH, session_started_at, fps, session_min_fps, session_max_fps)
                last_fps_summary_time = current_time

            if video_display_enabled:
                latest_plate_info = (
                    ocr_worker.get_latest_plate_info() if ocr_worker is not None else None
                )
                stats_info = (
                    ocr_worker.get_stats_info() if ocr_worker is not None else default_ocr_stats
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
            FPS_LOG_FILE_PATH,
            session_started_at,
            fps if "fps" in locals() else 0.0,
            session_min_fps,
            session_max_fps,
        )
        print(f"[FPS] Active history file: {active_history_label(FPS_LOG_FILE_PATH)}")
    finally:
        if ocr_worker is not None:
            ocr_worker.stop()
            ocr_worker.join(timeout=2)
        if mqtt_trigger is not None:
            mqtt_trigger.disconnect()
        frame_grabber.stop()
        frame_grabber.join(timeout=2)
        if video_display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
