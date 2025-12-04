import cv2
import numpy as np
import re


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


VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorbike",
    "motorcycle",
    "bicycle",
    "van",
}
