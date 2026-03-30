import time
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


def write_fps_summary(summary_path, session_started_at, current_fps, session_min_fps, session_max_fps):
    lines = [
        f"session_started_at={session_started_at}",
        f"current_fps={current_fps:.1f}",
        f"min_fps={session_min_fps:.1f}",
        f"max_fps={session_max_fps:.1f}",
        f"active_history_file={active_history_label(summary_path)}",
    ]
    atomic_write_text(summary_path, "\n".join(lines) + "\n")


def load_models():
    base_dir = Path(__file__).resolve().parent
    detector = YOLO(str(base_dir / "models" / "yolov8n_ncnn_model"), task="detect")
    plate_detector = YOLO(str(base_dir / "models" / "license_plate_detector.pt"), task="detect")
    return detector, plate_detector


def main():
    base_dir = Path(__file__).resolve().parent
    log_file_path = base_dir / "data" / "detected_plates.txt"
    fps_log_file_path = base_dir / "data" / "fps_stats.txt"
    fps_log_interval = 10.0
    fps_summary_interval = 1.0
    session_started_at = current_local_timestamp()

    # rtsp_url = "rtsp://user:password@192.168.1.50:554/Streaming/Channels/101"
    rtsp_url = "rtsp://192.168.1.196:554/stream1"

    print("starting RTSP stream...")
    frame_grabber = FrameGrabber(rtsp_url)
    ocr_worker = None

    try:
        frame_grabber.start()
        frame_grabber.ready.wait(timeout=5)
        if not frame_grabber.ready.is_set():
            raise RuntimeError("La capture RTSP ne repond pas.")
        if frame_grabber.last_error:
            raise frame_grabber.last_error

        print("Configuring display window...")
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 1280, 720)

        print("Reading initial frame...")
        frame = frame_grabber.get_latest(timeout=2)
        if frame is None:
            raise RuntimeError("Impossible de recuperer une frame initiale depuis le flux RTSP.")

        print("Loading YOLO models...")
        model, license_plate_detector_model = load_models()

        ocr_worker = PlateOcrWorker(log_file_path)
        ocr_worker.start()

        fps_history = []
        session_min_fps = 0.0
        session_max_fps = 0.0
        last_fps_log_time = time.time()
        last_fps_summary_time = 0.0
        write_fps_summary(fps_log_file_path, session_started_at, 0.0, session_min_fps, session_max_fps)

        while True:
            start_time = time.time()

            frame = frame_grabber.get_latest()
            if frame is None:
                print("Frame manquante, attente du flux...")
                time.sleep(0.01)
                continue

            results = model.predict(frame, imgsz=320, verbose=False)[0]
            annotated_frame = results.plot()
            submitted_this_frame = False

            for box in results.boxes:
                if submitted_this_frame or not ocr_worker.ready_for_new_job():
                    break

                cls_id = int(box.cls[0]) if box.cls is not None else -1
                cls_name = results.names.get(cls_id, str(cls_id)) if hasattr(results, "names") else str(cls_id)
                if cls_name.lower() not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                cv2.imshow("Cropped Plate", plate_crop)

                refined_crop = analyze_with_second_model(plate_crop, license_plate_detector_model)
                if refined_crop.size == 0:
                    continue

                if ocr_worker.submit(refined_crop):
                    submitted_this_frame = True
                    cv2.imshow("Plate", refined_crop)

            loop_time = time.time() - start_time
            fps = 1.0 / loop_time if loop_time > 0 else 0.0

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

            latest_plate_info = ocr_worker.get_latest_plate_info()
            stats_info = ocr_worker.get_stats_info()
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

            annotated_frame = draw_fps_info(annotated_frame, fps, min_fps, max_fps)
            cv2.imshow("Camera", annotated_frame)

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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
