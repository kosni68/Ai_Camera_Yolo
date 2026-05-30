"""Lancement d'un entrainement OCR en arriere-plan (pour la webapp).

L'entrainement est long et lourd : on l'execute dans un thread + sous-processus, en
redirigeant toute la sortie vers un fichier de log que la webapp affiche en direct.
Un seul entrainement a la fois (verrou). Enchaine deux etapes :
  1. export du dataset  (python -m src.training.export_dataset)
  2. entrainement + export ONNX  (python -m src.training.train_ocr --run)
"""

import os
import subprocess
import sys
import threading
from datetime import datetime

from src.core.config import PROJECT_ROOT

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"


class TrainingRunner:
    def __init__(self, dataset_dir, output_dir, log_path):
        self.dataset_dir = os.fspath(dataset_dir)
        self.output_dir = os.fspath(output_dir)
        self.log_path = os.fspath(log_path)

        self._lock = threading.Lock()
        self._thread = None
        self.status = STATUS_IDLE
        self.message = ""
        self.started_at = None
        self.finished_at = None

    def is_running(self):
        return self.status == STATUS_RUNNING

    def start(self, epochs=150, batch_size=64):
        epochs = max(1, min(int(epochs), 5000))
        batch_size = max(1, min(int(batch_size), 1024))
        with self._lock:
            if self.status == STATUS_RUNNING:
                return False, "Un entrainement est deja en cours."
            self.status = STATUS_RUNNING
            self.message = ""
            self.started_at = datetime.now().isoformat(timespec="seconds")
            self.finished_at = None
            self._thread = threading.Thread(
                target=self._run, args=(epochs, batch_size), daemon=True
            )
            self._thread.start()
            return True, "Entrainement lance."

    def _run(self, epochs, batch_size):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        env = dict(os.environ)
        env.setdefault("KERAS_BACKEND", "tensorflow")
        env["PYTHONUNBUFFERED"] = "1"  # logs en direct dans le fichier

        try:
            with open(self.log_path, "w", encoding="utf-8") as log:

                def step(title, cmd):
                    log.write(f"\n=== {title} ===\n$ {' '.join(cmd)}\n\n")
                    log.flush()
                    completed = subprocess.run(
                        cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT)
                    )
                    return completed.returncode

                rc = step(
                    "1/2 Export du dataset",
                    [sys.executable, "-m", "src.training.export_dataset", "--out", self.dataset_dir],
                )
                if rc != 0:
                    self._finish(STATUS_ERROR, "Echec de l'export du dataset (voir logs).")
                    return

                rc = step(
                    "2/2 Entrainement + export ONNX",
                    [
                        sys.executable, "-m", "src.training.train_ocr",
                        "--dataset-dir", self.dataset_dir,
                        "--output-dir", self.output_dir,
                        "--epochs", str(epochs),
                        "--batch-size", str(batch_size),
                        "--run",
                    ],
                )
                if rc != 0:
                    self._finish(STATUS_ERROR, "Echec de l'entrainement (voir logs).")
                    return

            self._finish(STATUS_DONE, "Entrainement termine. Modele dans " + self.output_dir)
        except Exception as exc:
            self._finish(STATUS_ERROR, f"Erreur: {exc}")

    def _finish(self, status, message):
        self.status = status
        self.message = message
        self.finished_at = datetime.now().isoformat(timespec="seconds")

    def read_log(self, max_bytes=40000):
        if not os.path.isfile(self.log_path):
            return ""
        with open(self.log_path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()[-max_bytes:]

    def state(self):
        return {
            "status": self.status,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
