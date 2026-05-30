"""Backend d'inference fast-plate-ocr (modele OCR entraine par l'utilisateur).

Charge un modele ONNX + son plate_config.yaml et lit le texte d'un crop de plaque.
La dependance est optionnelle (pip install 'fast-plate-ocr[onnx]') : on importe a la
demande pour ne rien imposer au runtime principal.

API verifiee sur fast-plate-ocr 1.1.0 :
  LicensePlateRecognizer(onnx_model_path=, plate_config_path=, device=).run(array, return_confidence=True)
  -> [PlatePrediction(plate: str, char_probs: np.ndarray | None, ...)]
Les arrays NumPy doivent deja etre dans le color mode du modele (grayscale ou rgb).
"""

import os

import cv2
import numpy as np

try:
    from fast_plate_ocr import LicensePlateRecognizer

    _FAST_PLATE_AVAILABLE = True
except ImportError:
    _FAST_PLATE_AVAILABLE = False


class FastPlateOcrBackend:
    """Enrobe LicensePlateRecognizer pour lire un crop OpenCV (BGR)."""

    def __init__(self, onnx_model_path, plate_config_path, device="cpu"):
        if not _FAST_PLATE_AVAILABLE:
            raise RuntimeError(
                "fast-plate-ocr n'est pas installe. Lance: pip install 'fast-plate-ocr[onnx]'"
            )

        onnx_model_path = os.fspath(onnx_model_path)
        plate_config_path = os.fspath(plate_config_path)
        if not os.path.isfile(onnx_model_path) or not os.path.isfile(plate_config_path):
            raise RuntimeError(
                f"Modele/config fast-plate-ocr introuvable: {onnx_model_path} / {plate_config_path}"
            )

        self.recognizer = LicensePlateRecognizer(
            onnx_model_path=onnx_model_path,
            plate_config_path=plate_config_path,
            device=device,
        )
        self.color_mode = getattr(self.recognizer.config, "image_color_mode", "grayscale")

    def _prepare(self, image):
        """Convertit le crop BGR OpenCV vers le color mode attendu par le modele."""
        if image is None or getattr(image, "size", 0) == 0:
            return None
        if self.color_mode == "grayscale":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        # rgb
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def read(self, image):
        """Lit le crop. Renvoie {raw, score, char_scores} ou None.

        `raw` est le texte brut decode ; le matching format FR et la normalisation
        sont faits en aval par le worker (via _build_candidate).
        """
        prepared = self._prepare(image)
        if prepared is None:
            return None

        predictions = self.recognizer.run(prepared, return_confidence=True)
        if not predictions:
            return None

        prediction = predictions[0]
        text = (prediction.plate or "").strip()
        if not text:
            return None

        score = 0.5
        char_scores = None
        raw_probs = getattr(prediction, "char_probs", None)
        if raw_probs is not None:
            probs = [float(value) for value in np.asarray(raw_probs).ravel()]
            if probs:
                # char_probs couvre les slots du modele ; on aligne sur le texte decode.
                if len(probs) >= len(text):
                    char_scores = probs[: len(text)]
                else:
                    char_scores = probs + [probs[-1]] * (len(text) - len(probs))
                score = sum(char_scores) / len(char_scores)

        return {"raw": text, "score": score, "char_scores": char_scores}
