import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

REQUIRED_CONFIG_KEYS = (
    "rtsp_url",
    "video_display_enabled",
    "overlay_enabled",
    "ocr_enabled",
    "ocr_fast_mode_enabled",
    "ocr_submit_interval_sec",
    "ocr_same_crop_retry_sec",
    "secondary_plate_detector_enabled",
    "secondary_plate_detector_model_path",
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


def load_runtime_config(config_path=None):
    config_path = Path(config_path) if config_path else CONFIG_PATH

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

    for bool_key in (
        "video_display_enabled",
        "overlay_enabled",
        "ocr_enabled",
        "ocr_fast_mode_enabled",
        "secondary_plate_detector_enabled",
        "save_detections_enabled",
        "roi_enabled",
    ):
        if not isinstance(raw_config[bool_key], bool):
            raise RuntimeError(f"Configuration key '{bool_key}' must be a boolean.")

    motion_detection_enabled = raw_config.get("motion_detection_enabled", False)
    if not isinstance(motion_detection_enabled, bool):
        raise RuntimeError("Configuration key 'motion_detection_enabled' must be a boolean.")

    secondary_plate_detector_model_path_value = str(
        raw_config["secondary_plate_detector_model_path"]
    ).strip()
    if not secondary_plate_detector_model_path_value:
        raise RuntimeError(
            "Configuration key 'secondary_plate_detector_model_path' must not be empty."
        )

    detection_save_root_value = str(raw_config["detection_save_root"]).strip()
    if not detection_save_root_value:
        raise RuntimeError("Configuration key 'detection_save_root' must not be empty.")

    try:
        detection_save_min_confidence = float(raw_config["detection_save_min_confidence"])
        fps_limit = float(raw_config["fps_limit"])
        fps_log_interval = float(raw_config["fps_log_interval"])
        fps_summary_interval = float(raw_config["fps_summary_interval"])
        detector_fps_limit = float(raw_config.get("detector_fps_limit", min(fps_limit, 10.0)))
        ocr_submit_interval_sec = float(raw_config["ocr_submit_interval_sec"])
        ocr_same_crop_retry_sec = float(raw_config["ocr_same_crop_retry_sec"])
        roi_x = float(raw_config["roi_x"])
        roi_y = float(raw_config["roi_y"])
        roi_width = float(raw_config["roi_width"])
        roi_height = float(raw_config["roi_height"])
        motion_resize_width = int(raw_config.get("motion_resize_width", 320))
        motion_diff_threshold = int(raw_config.get("motion_diff_threshold", 25))
        motion_min_area_ratio = float(raw_config.get("motion_min_area_ratio", 0.015))
        motion_keepalive_sec = float(raw_config.get("motion_keepalive_sec", 1.0))
        motion_force_detector_interval_sec = float(
            raw_config.get("motion_force_detector_interval_sec", 2.0)
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Configuration keys 'detection_save_min_confidence', 'fps_limit', "
            "'fps_log_interval', 'fps_summary_interval', 'detector_fps_limit', "
            "'ocr_submit_interval_sec', 'ocr_same_crop_retry_sec', "
            "'roi_x', 'roi_y', 'roi_width', 'roi_height', 'motion_resize_width', "
            "'motion_diff_threshold', 'motion_min_area_ratio', 'motion_keepalive_sec', "
            "and 'motion_force_detector_interval_sec' must be numeric."
        ) from exc

    if fps_limit <= 0:
        raise RuntimeError("Configuration key 'fps_limit' must be greater than 0.")
    if detector_fps_limit <= 0:
        raise RuntimeError("Configuration key 'detector_fps_limit' must be greater than 0.")
    if ocr_submit_interval_sec <= 0:
        raise RuntimeError("Configuration key 'ocr_submit_interval_sec' must be greater than 0.")
    if ocr_same_crop_retry_sec <= 0:
        raise RuntimeError("Configuration key 'ocr_same_crop_retry_sec' must be greater than 0.")
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
    if motion_resize_width < 32:
        raise RuntimeError("Configuration key 'motion_resize_width' must be >= 32.")
    if not (1 <= motion_diff_threshold <= 255):
        raise RuntimeError("Configuration key 'motion_diff_threshold' must be between 1 and 255.")
    if not (0.0 < motion_min_area_ratio <= 1.0):
        raise RuntimeError("Configuration key 'motion_min_area_ratio' must be between 0.0 and 1.0.")
    if motion_keepalive_sec <= 0:
        raise RuntimeError("Configuration key 'motion_keepalive_sec' must be greater than 0.")
    if motion_force_detector_interval_sec <= 0:
        raise RuntimeError(
            "Configuration key 'motion_force_detector_interval_sec' must be greater than 0."
        )

    # Resolve relative paths against PROJECT_ROOT (not the config file location).
    secondary_plate_detector_model_path = Path(secondary_plate_detector_model_path_value)
    if not secondary_plate_detector_model_path.is_absolute():
        secondary_plate_detector_model_path = PROJECT_ROOT / secondary_plate_detector_model_path

    detection_save_root = Path(detection_save_root_value)
    if not detection_save_root.is_absolute():
        detection_save_root = PROJECT_ROOT / detection_save_root

    mqtt_enabled = raw_config.get("mqtt_enabled", False)
    if not isinstance(mqtt_enabled, bool):
        raise RuntimeError("Configuration key 'mqtt_enabled' must be a boolean.")

    mqtt_broker_host = str(raw_config.get("mqtt_broker_host", "")).strip()
    mqtt_broker_port = int(raw_config.get("mqtt_broker_port", 1883))
    mqtt_username = str(raw_config.get("mqtt_username", ""))
    mqtt_password = str(raw_config.get("mqtt_password", ""))
    shelly_device_id = str(raw_config.get("shelly_device_id", "")).strip()
    shelly_pulse_duration_sec = float(raw_config.get("shelly_pulse_duration_sec", 3.0))

    if mqtt_enabled:
        if not mqtt_broker_host:
            raise RuntimeError("Configuration key 'mqtt_broker_host' must not be empty when mqtt_enabled=true.")
        if not shelly_device_id:
            raise RuntimeError("Configuration key 'shelly_device_id' must not be empty when mqtt_enabled=true.")
        if shelly_pulse_duration_sec <= 0:
            raise RuntimeError("Configuration key 'shelly_pulse_duration_sec' must be greater than 0.")

    registered_plates_path_value = str(raw_config.get("registered_plates_path", "config/registered_plates.json")).strip()
    registered_plates_path = Path(registered_plates_path_value)
    if not registered_plates_path.is_absolute():
        registered_plates_path = PROJECT_ROOT / registered_plates_path

    effective_detector_fps_limit = min(detector_fps_limit, fps_limit)
    detector_interval = 1.0 / effective_detector_fps_limit

    return {
        "rtsp_url": rtsp_url,
        "video_display_enabled": raw_config["video_display_enabled"],
        "overlay_enabled": raw_config["overlay_enabled"],
        "ocr_enabled": raw_config["ocr_enabled"],
        "ocr_fast_mode_enabled": raw_config["ocr_fast_mode_enabled"],
        "ocr_submit_interval_sec": ocr_submit_interval_sec,
        "ocr_same_crop_retry_sec": ocr_same_crop_retry_sec,
        "secondary_plate_detector_enabled": raw_config["secondary_plate_detector_enabled"],
        "secondary_plate_detector_model_path": secondary_plate_detector_model_path,
        "save_detections_enabled": raw_config["save_detections_enabled"],
        "detection_save_min_confidence": detection_save_min_confidence,
        "detection_save_root": detection_save_root,
        "detector_fps_limit": effective_detector_fps_limit,
        "roi": {
            "enabled": raw_config["roi_enabled"],
            "x": roi_x,
            "y": roi_y,
            "width": roi_width,
            "height": roi_height,
        },
        "motion": {
            "enabled": motion_detection_enabled,
            "resize_width": motion_resize_width,
            "diff_threshold": motion_diff_threshold,
            "min_area_ratio": motion_min_area_ratio,
            "keepalive_sec": max(motion_keepalive_sec, detector_interval),
            "force_detector_interval_sec": max(
                motion_force_detector_interval_sec,
                detector_interval,
            ),
        },
        "fps_limit": fps_limit,
        "fps_log_interval": fps_log_interval,
        "fps_summary_interval": fps_summary_interval,
        "mqtt": {
            "enabled": mqtt_enabled,
            "broker_host": mqtt_broker_host,
            "broker_port": mqtt_broker_port,
            "username": mqtt_username,
            "password": mqtt_password,
            "shelly_device_id": shelly_device_id,
            "pulse_duration_sec": shelly_pulse_duration_sec,
        },
        "registered_plates_path": registered_plates_path,
    }


def display_server_available():
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def adapt_runtime_config_for_environment(config):
    adjusted_config = dict(config)
    adjusted_config["roi"] = dict(config["roi"])
    adjusted_config["motion"] = dict(config["motion"])

    if adjusted_config["video_display_enabled"] and not display_server_available():
        adjusted_config["video_display_enabled"] = False
        adjusted_config["overlay_enabled"] = False
        print(
            "[CONFIG] No graphical display detected "
            "(DISPLAY/WAYLAND_DISPLAY missing). "
            "Disabling OpenCV windows and overlays for this run."
        )

    return adjusted_config
