import time
import cv2
from ultralytics import YOLO

from capture import FrameGrabber
from detection import analyze_with_second_model
from ocr_worker import PlateOcrWorker
from utils import draw_fps_info, VEHICLE_CLASSES


def load_models():
    # model principal
    detector = YOLO("./yolo_with_stream/models/yolov8n_ncnn_model")
    # second modèle pour raffiner la plaque
    plate_detector = YOLO("./yolo_with_stream/models/license_plate_detector.pt")
    return detector, plate_detector


def main():
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
    cv2.resizeWindow("Camera", 1280, 720)

    print("Reading initial frame...")
    frame = frame_grabber.get_latest(timeout=2)
    if frame is None:
        raise RuntimeError("Impossible de recuperer une frame initiale depuis le flux RTSP.")

    print("Loading YOLO models...")
    model, license_plate_detector_model = load_models()

    log_file_path = "./yolo_with_stream/data/detected_plates.txt"
    fps_log_file_path = "./yolo_with_stream/data/fps_stats.txt"
    fps_log_interval = 10
    last_fps_log_time = time.time()

    ocr_worker = PlateOcrWorker(log_file_path)
    ocr_worker.start()

    fps_history = []

    with open(fps_log_file_path, "a") as fps_log_file:
        while True:
            start_time = time.time()

            frame = frame_grabber.get_latest()
            if frame is None:
                print("Frame manquante, attente du flux...")
                time.sleep(0.01)
                continue

            print("Running YOLO prediction")
            results = model.predict(frame, imgsz=320)[0]
            annotated_frame = results.plot()

            for box in results.boxes:
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

                ocr_worker.submit(refined_crop)
                cv2.imshow("Plate", refined_crop)

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

            latest_plate = ocr_worker.get_latest_plate()
            if latest_plate:
                cv2.putText(
                    annotated_frame,
                    f"Derniere plaque: {latest_plate}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                )

            annotated_frame = draw_fps_info(annotated_frame, fps, min_fps, max_fps)
            cv2.imshow("Camera", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    ocr_worker.stop()
    ocr_worker.join(timeout=2)
    frame_grabber.stop()
    frame_grabber.join(timeout=2)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
