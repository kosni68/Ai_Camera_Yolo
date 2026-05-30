"""Lancement d'un entrainement OCR en arriere-plan (pour la webapp).

L'entrainement est long et lourd : on l'execute dans un thread + sous-processus, en
redirigeant toute la sortie vers un fichier de log que la webapp affiche en direct.
Un seul entrainement a la fois (verrou). Enchaine deux etapes :
  1. export du dataset  (python -m src.training.export_dataset)
  2. entrainement + export ONNX  (python -m src.training.train_ocr --run)
"""

import csv
import glob
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

    def list_models(self):
        """Liste les modeles OCR prets a tester/deployer (entraines + d'origine).

        Deux sources, sans copie manuelle de fichiers :
          - les entrainements : <output_dir>/<run>/best.onnx + plate_config.yaml adjacent ;
          - les modeles d'origine : <models>/*.onnx fournis avec le projet, avec leur
            config compagnon (ex: fast_plate_ocr.onnx + fast_plate_ocr_config.yaml).
        Chaque entree expose les chemins absolus (inference) et relatifs (config.json),
        la val_acc finale si dispo, et `kind` ('trained' ou 'origin'). Trie : entraines
        du plus recent au plus ancien, puis modeles d'origine.
        """
        models = []
        seen = set()

        def _add(name, kind, onnx_path, config_path, val_acc):
            key = os.path.normcase(os.path.abspath(onnx_path))
            if key in seen:
                return
            seen.add(key)
            models.append(
                {
                    "name": name,
                    "kind": kind,
                    "model_path": os.path.abspath(onnx_path),
                    "config_path": os.path.abspath(config_path),
                    "model_rel": _rel_to_root(onnx_path),
                    "config_rel": _rel_to_root(config_path),
                    "mtime": os.path.getmtime(onnx_path),
                    "val_acc": val_acc,
                }
            )

        trained = []
        for onnx_path in glob.glob(os.path.join(self.output_dir, "*", "best.onnx")):
            run_dir = os.path.dirname(onnx_path)
            config_path = os.path.join(run_dir, "plate_config.yaml")
            if not os.path.isfile(config_path):
                continue
            trained.append((os.path.getmtime(onnx_path), run_dir, onnx_path, config_path))
        for _, run_dir, onnx_path, config_path in sorted(trained, reverse=True):
            _add(
                os.path.basename(run_dir),
                "trained",
                onnx_path,
                config_path,
                _read_final_val_acc(os.path.join(run_dir, "training_log.csv")),
            )

        # Modeles d'origine fournis avec le projet (racine models/, hors ocr_training).
        models_root = os.path.dirname(self.output_dir)
        for onnx_path in sorted(glob.glob(os.path.join(models_root, "*.onnx"))):
            config_path = _find_companion_config(onnx_path)
            if config_path is None:
                continue
            name = os.path.splitext(os.path.basename(onnx_path))[0]
            _add(name, "origin", onnx_path, config_path, None)

        return models


def _rel_to_root(path):
    """Chemin relatif au projet (slashes /) si possible, sinon chemin absolu."""
    try:
        from pathlib import Path

        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return os.path.abspath(path)


def _find_companion_config(onnx_path):
    """Config YAML accompagnant un .onnx d'origine, ou None si aucune trouvee.

    Cherche, dans le meme dossier : <name>_config.yaml, <name>.yaml, plate_config.yaml.
    """
    base = os.path.splitext(onnx_path)[0]
    folder = os.path.dirname(onnx_path)
    for candidate in (
        base + "_config.yaml",
        base + ".yaml",
        os.path.join(folder, "plate_config.yaml"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _read_final_val_acc(log_path):
    """val_acc de la derniere epoch du training_log.csv, ou None si indisponible."""
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8", newline="") as handle:
            last = None
            for row in csv.DictReader(handle):
                last = row
            if last and last.get("val_acc") not in (None, ""):
                return float(last["val_acc"])
    except (OSError, ValueError, KeyError):
        return None
    return None
