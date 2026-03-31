import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from capture import FrameGrabber
from detection import analyze_with_second_model
from ocr_worker import configure_tesseract, create_easyocr_reader, run_easyocr_ocr, run_tesseract_ocr
from plate_recognition_tesseract import (
    BASE_DIR,
    CONFIG_PATH,
    box_intersects_roi,
    build_roi_pixels,
    crop_frame_to_roi,
    draw_detected_boxes,
    extract_detection_records,
    load_main_detector,
    load_plate_detector,
    load_runtime_config,
)
from utils import VEHICLE_CLASSES

DEFAULT_SCENARIOS = (
    "capture_only",
    "overlay_only",
    "detector_baseline",
    "detector_no_roi",
    "detector_low_rate_2fps",
    "save_frame_cost",
    "plate_pipeline",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CPU du pipeline RTSP live.")
    parser.add_argument("--duration", type=float, default=10.0, help="Duree de mesure par scenario.")
    parser.add_argument("--warmup", type=float, default=3.0, help="Duree de warmup exclue des mesures.")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=DEFAULT_SCENARIOS,
        help="Sous-ensemble de scenarios a executer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Dossier de sortie. Par defaut: data/benchmarks/<timestamp>.",
    )
    return parser.parse_args()


def build_output_dir(output_dir):
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_output_dir = BASE_DIR / "data" / "benchmarks" / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return run_output_dir


def create_result(name):
    return {
        "scenario": name,
        "status": "runtime_error",
        "wall_fps": 0.0,
        "process_cpu_pct": 0.0,
        "frames_received": 0,
        "detector_runs": 0,
        "vehicle_crops": 0,
        "ocr_jobs": 0,
        "files_written": 0,
        "main_detector_avg_ms": 0.0,
        "main_detector_p95_ms": 0.0,
        "second_detector_avg_ms": 0.0,
        "ocr_avg_ms": 0.0,
        "save_avg_ms": 0.0,
    }


def average(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def finalize_result(result, state, wall_elapsed, cpu_elapsed):
    result["wall_fps"] = result["frames_received"] / wall_elapsed if wall_elapsed > 0 else 0.0
    result["process_cpu_pct"] = cpu_elapsed / wall_elapsed * 100.0 if wall_elapsed > 0 else 0.0
    result["main_detector_avg_ms"] = average(state.get("main_detector_times_ms", []))
    result["main_detector_p95_ms"] = percentile(state.get("main_detector_times_ms", []), 0.95)
    result["second_detector_avg_ms"] = average(state.get("second_detector_times_ms", []))
    result["ocr_avg_ms"] = average(state.get("ocr_times_ms", []))
    result["save_avg_ms"] = average(state.get("save_times_ms", []))


def start_frame_grabber(rtsp_url):
    frame_grabber = FrameGrabber(rtsp_url, queue_size=1)
    frame_grabber.start()
    frame_grabber.ready.wait(timeout=5)
    if not frame_grabber.ready.is_set():
        raise RuntimeError("La capture RTSP ne repond pas.")
    if frame_grabber.last_error:
        raise frame_grabber.last_error
    return frame_grabber


def warmup_loop(frame_grabber, warmup_sec, process_frame, state, result):
    warmup_end = time.perf_counter() + max(0.0, warmup_sec)
    while time.perf_counter() < warmup_end:
        frame = frame_grabber.get_latest(timeout=1.0)
        if frame is None:
            continue
        process_frame(frame, state, result, record=False)


def measurement_loop(frame_grabber, duration_sec, process_frame, state, result):
    measure_started_at = time.perf_counter()
    cpu_started_at = time.process_time()

    while (time.perf_counter() - measure_started_at) < duration_sec:
        frame = frame_grabber.get_latest(timeout=1.0)
        if frame is None:
            continue

        result["frames_received"] += 1
        process_frame(frame, state, result, record=True)

    wall_elapsed = time.perf_counter() - measure_started_at
    cpu_elapsed = time.process_time() - cpu_started_at
    return wall_elapsed, cpu_elapsed


def make_detector_state(config):
    return {
        "config": config,
        "detector": load_main_detector(),
        "detector_interval": 1.0 / config["detector_fps_limit"],
        "last_detector_run_at": 0.0,
        "last_detections": [],
        "main_detector_times_ms": [],
    }


def run_main_detector(frame, state, result, record):
    config = state["config"]
    current_perf = time.perf_counter()
    fresh_detections = []
    roi_pixels = build_roi_pixels(frame, config["roi"])

    if state["last_detector_run_at"] == 0.0 or (current_perf - state["last_detector_run_at"]) >= state["detector_interval"]:
        detection_frame = crop_frame_to_roi(frame, roi_pixels)
        infer_started_at = time.perf_counter()
        results = state["detector"].predict(detection_frame, imgsz=320, verbose=False)[0]
        infer_ms = (time.perf_counter() - infer_started_at) * 1000.0

        x_offset = roi_pixels["x1"] if roi_pixels is not None else 0
        y_offset = roi_pixels["y1"] if roi_pixels is not None else 0
        fresh_detections = extract_detection_records(results, x_offset=x_offset, y_offset=y_offset)
        if roi_pixels is not None:
            fresh_detections = [
                detection for detection in fresh_detections if box_intersects_roi(detection, roi_pixels)
            ]

        state["last_detections"] = fresh_detections
        state["last_detector_run_at"] = time.perf_counter()

        if record:
            result["detector_runs"] += 1
            result["vehicle_crops"] += sum(
                1 for detection in fresh_detections if detection["label"] in VEHICLE_CLASSES
            )
            state["main_detector_times_ms"].append(infer_ms)

    return fresh_detections, roi_pixels


def create_ocr_backend():
    tesseract_path = configure_tesseract()
    easyocr_reader = create_easyocr_reader()
    if easyocr_reader is None and tesseract_path is None:
        raise RuntimeError("Aucun backend OCR disponible (EasyOCR/Tesseract).")
    return {
        "tesseract_path": tesseract_path,
        "easyocr_reader": easyocr_reader,
    }


def run_ocr_backend(image, ocr_backend):
    easyocr_reader = ocr_backend["easyocr_reader"]
    if easyocr_reader is not None:
        result = run_easyocr_ocr(image, easyocr_reader)
        if result is not None:
            return result

    if ocr_backend["tesseract_path"] is not None:
        return run_tesseract_ocr(image)

    return None


def process_capture_only(frame, state, result, record):
    return None


def process_overlay_only(frame, state, result, record):
    roi_pixels = build_roi_pixels(frame, state["config"]["roi"])
    overlay = draw_detected_boxes(frame, (), roi_pixels=roi_pixels)
    cv2.putText(
        overlay,
        "overlay_only",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 200, 255),
        2,
    )
    return None


def process_detector_only(frame, state, result, record):
    run_main_detector(frame, state, result, record)
    return None


def process_save_frame_cost(frame, state, result, record):
    if not record:
        return None

    current_perf = time.perf_counter()
    if (current_perf - state["last_save_at"]) < state["save_interval_sec"]:
        return None

    roi_pixels = build_roi_pixels(frame, state["config"]["roi"])
    annotated = draw_detected_boxes(frame, (), roi_pixels=roi_pixels)
    cv2.putText(
        annotated,
        "save_frame_cost",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 200, 255),
        2,
    )

    output_path = state["scenario_dir"] / f"frame-{state['save_index']:03d}.jpg"
    save_started_at = time.perf_counter()
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Impossible d'ecrire {output_path}")

    state["save_times_ms"].append((time.perf_counter() - save_started_at) * 1000.0)
    state["save_index"] += 1
    state["last_save_at"] = current_perf
    result["files_written"] += 1
    return None


def process_plate_pipeline(frame, state, result, record):
    fresh_detections, _ = run_main_detector(frame, state, result, record)
    if not fresh_detections:
        return None

    for detection in fresh_detections:
        if detection["label"] not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        second_detector_started_at = time.perf_counter()
        refined_crop = analyze_with_second_model(plate_crop, state["plate_detector"])
        second_detector_ms = (time.perf_counter() - second_detector_started_at) * 1000.0
        if record:
            state["second_detector_times_ms"].append(second_detector_ms)

        if refined_crop.size == 0:
            continue

        ocr_started_at = time.perf_counter()
        run_ocr_backend(refined_crop, state["ocr_backend"])
        ocr_ms = (time.perf_counter() - ocr_started_at) * 1000.0
        if record:
            state["ocr_times_ms"].append(ocr_ms)
            result["ocr_jobs"] += 1
        break

    return None


def build_scenario_state(name, base_config, run_output_dir):
    config = copy.deepcopy(base_config)

    if name == "capture_only":
        return config, {"config": config}, process_capture_only

    if name == "overlay_only":
        return config, {"config": config}, process_overlay_only

    if name == "detector_baseline":
        return config, make_detector_state(config), process_detector_only

    if name == "detector_no_roi":
        config["roi"]["enabled"] = False
        return config, make_detector_state(config), process_detector_only

    if name == "detector_low_rate_2fps":
        config["detector_fps_limit"] = 2.0
        return config, make_detector_state(config), process_detector_only

    if name == "save_frame_cost":
        scenario_dir = run_output_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        return config, {
            "config": config,
            "scenario_dir": scenario_dir,
            "last_save_at": 0.0,
            "save_interval_sec": 1.0,
            "save_index": 0,
            "save_times_ms": [],
        }, process_save_frame_cost

    if name == "plate_pipeline":
        state = make_detector_state(config)
        state["plate_detector"] = load_plate_detector()
        state["ocr_backend"] = create_ocr_backend()
        state["second_detector_times_ms"] = []
        state["ocr_times_ms"] = []
        return config, state, process_plate_pipeline

    raise ValueError(f"Scenario inconnu: {name}")


def run_scenario(name, base_config, warmup_sec, duration_sec, run_output_dir):
    print(f"[BENCH] Scenario: {name}")
    result = create_result(name)
    state = {}
    frame_grabber = None
    capture_ready = False

    try:
        _, state, process_frame = build_scenario_state(name, base_config, run_output_dir)
        frame_grabber = start_frame_grabber(base_config["rtsp_url"])
        capture_ready = True

        warmup_loop(frame_grabber, warmup_sec, process_frame, state, result)
        wall_elapsed, cpu_elapsed = measurement_loop(frame_grabber, duration_sec, process_frame, state, result)
        finalize_result(result, state, wall_elapsed, cpu_elapsed)

        if result["frames_received"] == 0:
            result["status"] = "capture_error"
        elif name == "plate_pipeline" and result["vehicle_crops"] == 0:
            result["status"] = "skipped_no_vehicle_crop"
        else:
            result["status"] = "ok"
    except Exception as exc:
        result["status"] = "runtime_error" if capture_ready else "capture_error"
        result["error"] = str(exc)
    finally:
        if frame_grabber is not None:
            frame_grabber.stop()
            frame_grabber.join(timeout=2)

    return result


def print_results_table(results):
    headers = (
        ("scenario", 24),
        ("status", 24),
        ("fps", 7),
        ("cpu%", 8),
        ("det", 5),
        ("veh", 5),
        ("ocr", 5),
        ("save", 5),
        ("det_ms", 8),
        ("det_p95", 8),
        ("plate_ms", 9),
        ("ocr_ms", 8),
        ("save_ms", 8),
    )

    def format_row(values):
        cells = []
        for (label, width), value in zip(headers, values):
            cells.append(f"{value:<{width}}")
        return " ".join(cells)

    print(format_row([label for label, _ in headers]))
    print(format_row(["-" * min(width, len(label)) for label, width in headers]))
    for item in results:
        print(
            format_row(
                [
                    item["scenario"],
                    item["status"],
                    f"{item['wall_fps']:.1f}",
                    f"{item['process_cpu_pct']:.1f}",
                    str(item["detector_runs"]),
                    str(item["vehicle_crops"]),
                    str(item["ocr_jobs"]),
                    str(item["files_written"]),
                    f"{item['main_detector_avg_ms']:.1f}",
                    f"{item['main_detector_p95_ms']:.1f}",
                    f"{item['second_detector_avg_ms']:.1f}",
                    f"{item['ocr_avg_ms']:.1f}",
                    f"{item['save_avg_ms']:.1f}",
                ]
            )
        )


def main():
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be greater than 0.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be greater than or equal to 0.")

    scenario_names = args.scenarios if args.scenarios else list(DEFAULT_SCENARIOS)
    output_dir = build_output_dir(args.output_dir)
    config = load_runtime_config(CONFIG_PATH)

    print(f"[BENCH] Output dir: {output_dir}")
    print(f"[BENCH] RTSP URL: {config['rtsp_url']}")
    print(f"[BENCH] Duration: {args.duration:.1f}s | Warmup: {args.warmup:.1f}s")

    results = []
    for name in scenario_names:
        results.append(run_scenario(name, config, args.warmup, args.duration, output_dir))

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rtsp_url": config["rtsp_url"],
        "duration_sec": args.duration,
        "warmup_sec": args.warmup,
        "output_dir": str(output_dir),
        "scenarios": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_results_table(results)
    print(f"[BENCH] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
