import re

import cv2
import numpy as np

STANDARD_PLATE_WIDTH = 520
STANDARD_PLATE_HEIGHT = 110

LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "T": "7",
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "7": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}

PLATE_PATTERNS = (
    ("LLDDDLL", lambda clean: f"{clean[:2]}-{clean[2:5]}-{clean[5:]}"),
    ("DDDDLLDD", lambda clean: f"{clean[:4]} {clean[4:6]} {clean[6:]}"),
    ("DDDLLDD", lambda clean: f"{clean[:3]} {clean[3:5]} {clean[5:]}"),
)


def normalize_ocr_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _candidate_windows(text, target_length):
    if len(text) < target_length:
        return []
    if len(text) == target_length:
        return [text]
    return [text[index:index + target_length] for index in range(len(text) - target_length + 1)]


def _coerce_to_pattern(text, pattern):
    if len(text) != len(pattern):
        return None

    coerced = []
    substitutions = 0

    for char, expected in zip(text, pattern):
        if expected == "L":
            if char.isalpha():
                coerced.append(char)
                continue
            mapped = DIGIT_TO_LETTER.get(char)
            if mapped is None:
                return None
            coerced.append(mapped)
            substitutions += 1
            continue

        if char.isdigit():
            coerced.append(char)
            continue
        mapped = LETTER_TO_DIGIT.get(char)
        if mapped is None:
            return None
        coerced.append(mapped)
        substitutions += 1

    return "".join(coerced), substitutions


def match_french_plate(text):
    clean = normalize_ocr_text(text)
    if not clean:
        return None

    matches = []
    for pattern, formatter in PLATE_PATTERNS:
        for window in _candidate_windows(clean, len(pattern)):
            candidate = _coerce_to_pattern(window, pattern)
            if candidate is None:
                continue
            coerced, substitutions = candidate
            matches.append((substitutions, abs(len(clean) - len(pattern)), coerced, formatter(coerced)))

    if not matches:
        return None

    matches.sort(key=lambda item: (item[0], item[1]))
    _, _, normalized, formatted = matches[0]
    return normalized, formatted


def is_french_plate(text):
    return match_french_plate(text) is not None


def format_french_plate(text):
    match = match_french_plate(text)
    if match is not None:
        _, formatted = match
        return formatted
    return text


def _ensure_grayscale(image):
    if image is None or image.size == 0:
        return np.array([], dtype=np.uint8)
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    point_sums = pts.sum(axis=1)
    rect[0] = pts[np.argmin(point_sums)]
    rect[2] = pts[np.argmax(point_sums)]

    point_diffs = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(point_diffs)]
    rect[3] = pts[np.argmax(point_diffs)]
    return rect


def _find_plate_corners(image):
    gray = _ensure_grayscale(image)
    if gray.size == 0:
        return None

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=2,
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = gray.shape[0] * gray.shape[1]
    best_corners = None
    best_score = None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.10:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype("float32")
        else:
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype("float32")

        ordered = _order_points(corners)
        width_top = np.linalg.norm(ordered[1] - ordered[0])
        width_bottom = np.linalg.norm(ordered[2] - ordered[3])
        height_left = np.linalg.norm(ordered[3] - ordered[0])
        height_right = np.linalg.norm(ordered[2] - ordered[1])
        width = max(width_top, width_bottom)
        height = max(height_left, height_right, 1.0)
        aspect_ratio = width / height
        if aspect_ratio < 1.8:
            continue

        score = (area / image_area) - abs(aspect_ratio - 4.7) * 0.08
        if best_score is None or score > best_score:
            best_corners = ordered
            best_score = score

    return best_corners


def rectify_plate(image, output_width=STANDARD_PLATE_WIDTH, output_height=STANDARD_PLATE_HEIGHT):
    corners = _find_plate_corners(image)
    if corners is None:
        return image.copy()

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (output_width, output_height))


def crop_plate_text_region(image):
    if image is None or image.size == 0:
        return np.array([], dtype=np.uint8)

    height, width = image.shape[:2]
    y1 = int(height * 0.08)
    y2 = int(height * 0.92)
    x1 = int(width * 0.05)
    x2 = int(width * 0.95)
    return image[y1:y2, x1:x2]


def prepare_plate_for_ocr(image):
    rectified = rectify_plate(image)
    text_region = crop_plate_text_region(rectified)
    return rectified, text_region


def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def preprocess_for_ocr(image):
    _, text_region = prepare_plate_for_ocr(image)
    gray = _ensure_grayscale(text_region)
    if gray.size == 0:
        return gray

    denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blurred)


def build_ocr_variants(image):
    base = preprocess_for_ocr(image)
    if base.size == 0:
        return []

    sharpened = sharpen_image(base)
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    inverted = cv2.bitwise_not(otsu)

    variants = []
    for name, variant in (
        ("gray", base),
        ("sharpened", sharpened),
        ("otsu", otsu),
        ("adaptive", adaptive),
        ("inverse", inverted),
    ):
        upscaled = cv2.resize(variant, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        variants.append((name, upscaled))
    return variants
