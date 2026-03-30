import time
import threading
from queue import Queue, Empty
import os
import shutil
from collections import Counter, deque
import pytesseract
from pytesseract import TesseractNotFoundError
import cv2

try:
    import easyocr
except ImportError:
    easyocr = None

from utils import (
    build_ocr_variants,
    format_french_plate,
    is_french_plate,
    match_french_plate,
    normalize_ocr_text,
    prepare_plate_for_ocr,
)


TESSERACT_CONFIGS = (
    ("line", r"--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    ("word", r"--oem 3 --psm 8 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    ("block", r"--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    ("raw", r"--oem 3 --psm 13 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
)

EASYOCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def resolve_tesseract_path():
    for env_var in ("TESSERACT_CMD", "TESSERACT_PATH"):
        candidate = os.environ.get(env_var)
        if candidate and os.path.exists(candidate):
            return candidate

    for command_name in ("tesseract.exe", "tesseract"):
        candidate = shutil.which(command_name)
        if candidate:
            return candidate

    candidates = []
    if os.name == "nt":
        for base_dir in (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            r"C:\Program Files",
            r"C:\Program Files (x86)",
        ):
            if base_dir:
                candidates.append(os.path.join(base_dir, "Tesseract-OCR", "tesseract.exe"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def configure_tesseract():
    tesseract_path = resolve_tesseract_path()
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    return tesseract_path


def create_easyocr_reader():
    if easyocr is None:
        return None

    try:
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:
        print(f"[OCR] EasyOCR indisponible: {exc}")
        return None


def _build_candidate(raw_text, score, backend, char_scores=None, source=""):
    clean_text = normalize_ocr_text(raw_text)
    if not clean_text:
        return None

    match = match_french_plate(clean_text)
    normalized = None
    formatted = clean_text
    if match is not None:
        normalized, formatted = match

    if not char_scores:
        char_scores = [score] * len(clean_text)

    return {
        "raw": clean_text,
        "normalized": normalized,
        "formatted": formatted,
        "score": float(score),
        "char_scores": [float(value) for value in char_scores],
        "backend": backend,
        "source": source,
    }


def run_easyocr_ocr(image, reader):
    rectified_plate, text_region = prepare_plate_for_ocr(image)
    candidates = []

    for variant_name, variant in (("rectified", rectified_plate), ("text", text_region)):
        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist=EASYOCR_ALLOWLIST,
        )
        tokens = []
        token_scores = []
        char_scores = []
        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            text = item[1]
            confidence = item[2]
            clean_text = normalize_ocr_text(text)
            if not clean_text:
                continue
            token_score = max(0.05, float(confidence))
            tokens.append(clean_text)
            token_scores.append(token_score)
            char_scores.extend([token_score] * len(clean_text))

        if not tokens or not token_scores:
            continue

        combined = "".join(tokens)
        score = min(token_scores) + (sum(token_scores) / len(token_scores)) * 0.15
        candidate = _build_candidate(
            combined,
            score=score,
            backend="easyocr",
            char_scores=char_scores,
            source=variant_name,
        )
        if candidate is not None:
            candidates.append(candidate)

    if candidates:
        candidates.sort(
            key=lambda candidate: (
                0 if candidate["normalized"] else 1,
                -candidate["score"],
                -len(candidate["raw"]),
                0 if candidate["source"] == "rectified" else 1,
            )
        )
        best_candidate = candidates[0]
        if best_candidate["normalized"] is None:
            preview = ", ".join(
                f"{candidate['source']}={candidate['raw']}@{candidate['score']:.2f}"
                for candidate in candidates[:4]
            )
            print(f"[OCR] Tentatives EasyOCR: {preview}")
        return best_candidate

    return None


def run_tesseract_ocr(image):
    attempts = []
    best_candidate = None

    for variant_name, variant in build_ocr_variants(image):
        for config_name, config in TESSERACT_CONFIGS:
            text = pytesseract.image_to_string(variant, config=config).strip()
            candidate = _build_candidate(
                text,
                score=0.20 if is_french_plate(text) else 0.08,
                backend="tesseract",
                source=f"{variant_name}/{config_name}",
            )
            if candidate is None:
                continue

            attempts.append(f"{candidate['source']}={candidate['raw']}")
            if candidate["normalized"] is not None:
                return candidate

            if best_candidate is None or len(candidate["raw"]) > len(best_candidate["raw"]):
                best_candidate = candidate

    if attempts:
        preview = ", ".join(attempts[:6])
        print(f"[OCR] Tentatives: {preview}")

    return best_candidate


def run_ocr(image, easyocr_reader=None):
    if easyocr_reader is not None:
        result = run_easyocr_ocr(image, easyocr_reader)
        if result is not None:
            return result

    return run_tesseract_ocr(image)


class PlateOcrWorker(threading.Thread):
    """Traite l'OCR en asynchrone pour ne pas bloquer l'inference principale."""

    def __init__(self, log_file_path, queue_size=5):
        super().__init__(daemon=True)
        self.queue = Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.log_file_path = log_file_path
        self.last_plate = None
        self.latest_text = None
        self.lock = threading.Lock()
        self.recent_candidates = deque(maxlen=10)
        self.last_submit_time = 0.0
        self.last_signature = None
        self.submit_interval_sec = 0.35
        self.same_crop_retry_sec = 1.0
        self.signature_diff_threshold = 2.0
        self.tesseract_path = configure_tesseract()
        self.easyocr_reader = create_easyocr_reader()
        self.ocr_available = self.easyocr_reader is not None or self.tesseract_path is not None

        if self.easyocr_reader is not None:
            print("[OCR] EasyOCR active")
        if self.tesseract_path:
            print(f"[OCR] Using Tesseract at {self.tesseract_path}")
        if not self.ocr_available:
            print("[OCR] Tesseract introuvable. Definis TESSERACT_CMD ou relance setup_env.ps1.")

    def _compute_signature(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 12), interpolation=cv2.INTER_AREA)

    def ready_for_new_job(self):
        if not self.ocr_available:
            return False
        if self.queue.qsize() > 0:
            return False
        return (time.time() - self.last_submit_time) >= self.submit_interval_sec

    def submit(self, crop):
        if not self.ocr_available or crop is None or crop.size == 0:
            return False

        now = time.time()
        if (now - self.last_submit_time) < self.submit_interval_sec:
            return False

        signature = self._compute_signature(crop)
        if self.last_signature is not None:
            signature_diff = cv2.absdiff(signature, self.last_signature).mean()
            if signature_diff < self.signature_diff_threshold and (now - self.last_submit_time) < self.same_crop_retry_sec:
                return False

        if self.queue.full():
            try:
                self.queue.get_nowait()
            except Empty:
                pass
        self.last_submit_time = now
        self.last_signature = signature
        self.queue.put(crop.copy())
        return True

    def get_latest_plate(self):
        with self.lock:
            return self.latest_text

    def _build_consensus_plate(self):
        if len(self.recent_candidates) < 2:
            return None

        length_counts = Counter(len(candidate["normalized"]) for candidate in self.recent_candidates if candidate["normalized"])
        if not length_counts:
            return None

        target_length = length_counts.most_common(1)[0][0]
        candidates = [candidate for candidate in self.recent_candidates if candidate["normalized"] and len(candidate["normalized"]) == target_length]
        if len(candidates) < 2:
            return None

        per_position_votes = [Counter() for _ in range(target_length)]
        per_position_totals = [0.0] * target_length

        for candidate in candidates:
            normalized = candidate["normalized"]
            char_scores = candidate["char_scores"]
            default_score = candidate["score"]
            if len(char_scores) != len(normalized):
                char_scores = [default_score] * len(normalized)

            for index, char in enumerate(normalized):
                weight = char_scores[index] if index < len(char_scores) else default_score
                per_position_votes[index][char] += weight
                per_position_totals[index] += weight

        consensus_chars = []
        dominance = []
        for index, votes in enumerate(per_position_votes):
            if not votes:
                return None
            best_char, best_weight = max(votes.items(), key=lambda item: item[1])
            consensus_chars.append(best_char)
            total_weight = max(per_position_totals[index], 1e-6)
            dominance.append(best_weight / total_weight)

        consensus_normalized = "".join(consensus_chars)
        if self.last_plate:
            previous_normalized = normalize_ocr_text(self.last_plate)
            if len(previous_normalized) == len(consensus_normalized) and previous_normalized[:5] == consensus_normalized[:5]:
                merged_chars = list(consensus_normalized)
                for index, ratio in enumerate(dominance):
                    if ratio < 0.55:
                        merged_chars[index] = previous_normalized[index]
                consensus_normalized = "".join(merged_chars)

                differing_positions = [
                    index for index, (previous_char, current_char) in enumerate(zip(previous_normalized, consensus_normalized))
                    if previous_char != current_char
                ]
                if differing_positions and all(index >= 5 for index in differing_positions):
                    consensus_normalized = previous_normalized

        match = match_french_plate(consensus_normalized)
        if match is None:
            return None

        normalized, formatted = match
        return {
            "normalized": normalized,
            "formatted": formatted,
            "dominance": dominance,
            "confidence": sum(max(votes.values()) for votes in per_position_votes) / target_length,
        }

    def _update_stable_plate(self, candidate):
        self.recent_candidates.append(candidate)
        consensus = self._build_consensus_plate()

        with self.lock:
            if consensus is None:
                self.latest_text = candidate["formatted"]
                return None

            self.latest_text = consensus["formatted"]
            has_changed = consensus["formatted"] != self.last_plate
            required_samples = 6 if self.last_plate is None else 3
            required_dominance = 0.50 if self.last_plate is None else 0.45
            if len(self.recent_candidates) >= required_samples and min(consensus["dominance"]) >= required_dominance and has_changed:
                self.last_plate = consensus["formatted"]
                return consensus["formatted"]

        return None

    def run(self):
        if not self.ocr_available:
            return

        while not self.stop_event.is_set():
            try:
                crop = self.queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                result = run_ocr(crop, self.easyocr_reader)
            except TesseractNotFoundError:
                print("[OCR] Tesseract est installe mais n'a pas pu etre lance. Verifie le PATH ou TESSERACT_CMD.")
                if self.easyocr_reader is None:
                    self.ocr_available = False
                    break
            except Exception as exc:
                print(f"[OCR] Erreur pendant la lecture de plaque: {exc}")
                continue

            if result is None:
                continue

            if result["normalized"] is not None:
                stable_plate = self._update_stable_plate(result)
                if stable_plate is not None:
                    print(f"[OCR] Plaque stabilisee: {stable_plate}")
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(self.log_file_path, "a") as log_file:
                        log_file.write(f"[{timestamp}] {stable_plate}\n")
            else:
                print(f"[IGNORED] Non-French plate format: {result['raw']}")

    def stop(self):
        self.stop_event.set()
