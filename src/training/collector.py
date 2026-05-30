"""Collecte d'echantillons de plaques pour l'entrainement OCR.

Branche sur le worker OCR : a chaque crop traite, sauvegarde l'image dans un dossier
dedie et insere une ligne dans l'index SQLite avec la lecture OCR. Deux garde-fous
evitent de saturer le disque quand un vehicule stationne devant la camera :
  - dedup : on saute si la meme lecture revient dans `dedup_window_sec` ;
  - plafond : au plus `max_per_minute` echantillons par minute.

Le schema de nommage (sous-dossier par jour, horodatage a la milliseconde) reprend
celui de save_plate_image() pour rester coherent, mais en silencieux.
"""

import os
import threading
import time
from datetime import datetime

import cv2


class SampleCollector:
    def __init__(
        self,
        store,
        image_root,
        dedup_window_sec=8.0,
        max_per_minute=30,
        enabled=True,
    ):
        self.store = store
        self.image_root = os.fspath(image_root)
        self.dedup_window_sec = float(dedup_window_sec)
        self.max_per_minute = int(max_per_minute)
        self.enabled = enabled

        self._lock = threading.Lock()
        self._last_prediction = None
        self._last_collect_time = 0.0
        self._minute_window_start = 0.0
        self._minute_count = 0

    def _save_image(self, crop):
        timestamp = datetime.now()
        day_folder = timestamp.strftime("%Y-%m-%d")
        file_stamp = timestamp.strftime("%Y%m%d-%H%M%S-%f")[:-3]
        output_path = os.path.join(self.image_root, day_folder, f"{file_stamp}.jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, crop):
            return None
        return output_path

    def _should_collect(self, normalized_prediction, now):
        """Applique dedup + plafond. Doit etre appele sous le verrou."""
        if (
            normalized_prediction
            and normalized_prediction == self._last_prediction
            and (now - self._last_collect_time) < self.dedup_window_sec
        ):
            return False

        if (now - self._minute_window_start) >= 60.0:
            self._minute_window_start = now
            self._minute_count = 0
        if self._minute_count >= self.max_per_minute:
            return False

        self._minute_count += 1
        self._last_prediction = normalized_prediction
        self._last_collect_time = now
        return True

    def collect(self, crop, prediction=None, confidence=None, backend=None, source="ocr"):
        """Sauvegarde le crop + insere un echantillon. Renvoie l'id ou None si saute."""
        if not self.enabled:
            return None
        if crop is None or getattr(crop, "size", 0) == 0:
            return None

        now = time.time()
        normalized_prediction = (prediction or "").strip().upper()
        with self._lock:
            if not self._should_collect(normalized_prediction, now):
                return None

        image_path = self._save_image(crop)
        if image_path is None:
            return None

        return self.store.add_sample(
            image_path,
            ocr_prediction=prediction,
            ocr_confidence=confidence,
            source=source,
            ocr_backend=backend,
        )
