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
    is_french_plate,
    match_french_plate,
    normalize_ocr_text,
    prepare_plate_for_ocr,
)
from log_utils import (
    active_history_label,
    atomic_write_text,
    append_line,
    current_local_timestamp,
    daily_history_path,
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
        self.log_file_path = os.fspath(log_file_path)
        self.last_plate = None
        self.last_plate_count = 0
        self.latest_text = None
        self.latest_plate_info = None
        self.lock = threading.Lock()
        self.recent_candidates = deque(maxlen=10)
        self.last_submit_time = 0.0
        self.last_signature = None
        self.submit_interval_sec = 0.35
        self.same_crop_retry_sec = 1.0
        self.signature_diff_threshold = 2.0
        self.plate_history_interval_sec = 30.0
        self.last_history_plate = None
        self.last_history_write_time = 0.0
        self.last_history_label = None
        self.session_started_at = current_local_timestamp()
        self.ocr_jobs_processed = 0
        self.ocr_success_stabilized = 0
        self.ocr_failure_total = 0
        self.ocr_failure_non_french = 0
        self.ocr_failure_unstable = 0
        self.ocr_failure_empty = 0
        self.tesseract_path = None
        self.easyocr_reader = None
        self.ocr_available = True
        self.ocr_initialized = False
        self.ocr_init_lock = threading.Lock()

        self._write_compact_log()

    def _ensure_ocr_initialized(self):
        if self.ocr_initialized:
            return self.ocr_available

        with self.ocr_init_lock:
            if self.ocr_initialized:
                return self.ocr_available

            self.tesseract_path = configure_tesseract()
            self.easyocr_reader = create_easyocr_reader()
            self.ocr_available = self.easyocr_reader is not None or self.tesseract_path is not None
            self.ocr_initialized = True

            if self.easyocr_reader is not None:
                print("[OCR] EasyOCR active")
            if self.tesseract_path:
                print(f"[OCR] Using Tesseract at {self.tesseract_path}")
            if not self.ocr_available:
                print("[OCR] Tesseract introuvable. Definis TESSERACT_CMD ou relance setup_env.ps1.")

        return self.ocr_available

    def _compute_signature(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 12), interpolation=cv2.INTER_AREA)

    def ready_for_new_job(self):
        if self.ocr_initialized and not self.ocr_available:
            return False
        if self.queue.qsize() > 0:
            return False
        return (time.time() - self.last_submit_time) >= self.submit_interval_sec

    def submit(self, crop):
        if crop is None or crop.size == 0:
            return False

        if self.ocr_initialized and not self.ocr_available:
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
            if self.latest_plate_info is None:
                return None
            return self.latest_plate_info["plate"]

    def get_latest_plate_info(self):
        with self.lock:
            if self.latest_plate_info is None:
                return None
            return dict(self.latest_plate_info)

    def get_stats_info(self):
        with self.lock:
            return self._build_stats_info_locked()

    def _build_stats_info_locked(self):
        current_plate = None
        consecutive_reads = 0
        if self.latest_plate_info is not None:
            current_plate = self.latest_plate_info["plate"]
            consecutive_reads = self.latest_plate_info["consecutive_count"]

        processed = self.ocr_jobs_processed
        success_rate = (self.ocr_success_stabilized / processed * 100.0) if processed else 0.0
        failure_rate = (self.ocr_failure_total / processed * 100.0) if processed else 0.0

        return {
            "current_plate": current_plate,
            "consecutive_reads": consecutive_reads,
            "ocr_jobs_processed": processed,
            "ocr_success_stabilized": self.ocr_success_stabilized,
            "ocr_failure_total": self.ocr_failure_total,
            "ocr_failure_non_french": self.ocr_failure_non_french,
            "ocr_failure_unstable": self.ocr_failure_unstable,
            "ocr_failure_empty": self.ocr_failure_empty,
            "ocr_success_rate_pct": success_rate,
            "ocr_failure_rate_pct": failure_rate,
            "session_started_at": self.session_started_at,
            "active_history_file": active_history_label(self.log_file_path),
        }

    def _write_compact_log(self):
        with self.lock:
            stats = self._build_stats_info_locked()

        lines = [
            f"current_plate={stats['current_plate'] or ''}",
            f"consecutive_reads={stats['consecutive_reads']}",
            f"ocr_jobs_processed={stats['ocr_jobs_processed']}",
            f"ocr_success_stabilized={stats['ocr_success_stabilized']}",
            f"ocr_failure_total={stats['ocr_failure_total']}",
            f"ocr_failure_non_french={stats['ocr_failure_non_french']}",
            f"ocr_failure_unstable={stats['ocr_failure_unstable']}",
            f"ocr_failure_empty={stats['ocr_failure_empty']}",
            f"ocr_success_rate_pct={stats['ocr_success_rate_pct']:.1f}",
            f"ocr_failure_rate_pct={stats['ocr_failure_rate_pct']:.1f}",
            f"session_started_at={stats['session_started_at']}",
            f"active_history_file={stats['active_history_file']}",
        ]
        atomic_write_text(self.log_file_path, "\n".join(lines) + "\n")

    def _record_job_outcome(self, outcome):
        with self.lock:
            self.ocr_jobs_processed += 1
            if outcome == "success":
                self.ocr_success_stabilized += 1
            else:
                self.ocr_failure_total += 1
                if outcome == "non_french":
                    self.ocr_failure_non_french += 1
                elif outcome == "unstable":
                    self.ocr_failure_unstable += 1
                elif outcome == "empty":
                    self.ocr_failure_empty += 1
                else:
                    raise ValueError(f"Unknown OCR outcome: {outcome}")

        self._write_compact_log()

    def _maybe_append_plate_history(self, stable_plate_info):
        plate = stable_plate_info["plate"]
        consecutive_count = stable_plate_info["consecutive_count"]
        history_file = daily_history_path(self.log_file_path)
        history_label = active_history_label(self.log_file_path)
        now = time.time()
        timestamp = current_local_timestamp()

        should_write = False
        suffix = ""
        if self.last_history_plate != plate or self.last_history_label != history_label:
            should_write = True
        elif (now - self.last_history_write_time) >= self.plate_history_interval_sec:
            should_write = True
            suffix = f" | heartbeat={int(self.plate_history_interval_sec)}s"

        if not should_write:
            return

        append_line(
            history_file,
            f"[{timestamp}] {plate} | consecutive_reads={consecutive_count}{suffix}",
        )
        self.last_history_plate = plate
        self.last_history_write_time = now
        self.last_history_label = history_label

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
                return None

            required_samples = 6 if self.last_plate is None else 3
            required_dominance = 0.50 if self.last_plate is None else 0.45
            if len(self.recent_candidates) < required_samples or min(consensus["dominance"]) < required_dominance:
                return None

            if consensus["formatted"] == self.last_plate:
                self.last_plate_count += 1
            else:
                self.last_plate = consensus["formatted"]
                self.last_plate_count = 1

            self.latest_text = self.last_plate
            self.latest_plate_info = {
                "plate": self.last_plate,
                "consecutive_count": self.last_plate_count,
            }
            return dict(self.latest_plate_info)

    def run(self):
        while not self.stop_event.is_set():
            try:
                crop = self.queue.get(timeout=0.2)
            except Empty:
                continue

            if not self._ensure_ocr_initialized():
                continue

            result = None
            try:
                result = run_ocr(crop, self.easyocr_reader)
            except TesseractNotFoundError:
                print("[OCR] Tesseract est installe mais n'a pas pu etre lance. Verifie le PATH ou TESSERACT_CMD.")
                if self.easyocr_reader is None:
                    self.ocr_available = False
                    break
                self._record_job_outcome("empty")
                continue
            except Exception as exc:
                print(f"[OCR] Erreur pendant la lecture de plaque: {exc}")
                continue

            if result is None:
                self._record_job_outcome("empty")
                continue

            if result["normalized"] is not None:
                stable_plate_info = self._update_stable_plate(result)
                if stable_plate_info is not None:
                    plate = stable_plate_info["plate"]
                    consecutive_count = stable_plate_info["consecutive_count"]
                    print(f"[OCR] Plaque stabilisee: {plate} (x{consecutive_count})")
                    self._record_job_outcome("success")
                    self._maybe_append_plate_history(stable_plate_info)
                else:
                    self._record_job_outcome("unstable")
            else:
                print(f"[IGNORED] Non-French plate format: {result['raw']}")
                self._record_job_outcome("non_french")

    def stop(self):
        self.stop_event.set()
