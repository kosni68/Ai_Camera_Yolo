"""Index SQLite des echantillons de plaques collectes pour l'entrainement OCR.

Le pipeline (thread OCR) ecrit des echantillons en continu ; la webapp de validation
lit et met a jour les labels en parallele. SQLite en mode WAL + un verrou applicatif
suffisent largement vu le faible debit (quelques inserts/seconde au plus).

Chaque echantillon pointe vers une image de crop (chemin relatif a la racine projet)
et porte la lecture OCR brute, le label valide/corrige par l'humain, et un statut.
"""

import os
import sqlite3
import threading
from datetime import datetime

from src.core.config import PROJECT_ROOT

STATUS_PENDING = "pending"
STATUS_VALIDATED = "validated"
STATUS_CORRECTED = "corrected"
STATUS_REJECTED = "rejected"

VALID_STATUSES = (STATUS_PENDING, STATUS_VALIDATED, STATUS_CORRECTED, STATUS_REJECTED)
# Statuts qui portent un label exploitable pour construire le dataset d'entrainement.
LABELED_STATUSES = (STATUS_VALIDATED, STATUS_CORRECTED)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    image_path     TEXT NOT NULL,
    source         TEXT NOT NULL,
    ocr_prediction TEXT,
    ocr_confidence REAL,
    ocr_backend    TEXT,
    label          TEXT,
    status         TEXT NOT NULL DEFAULT 'pending',
    labeled_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status);
"""


def _now():
    return datetime.now().isoformat(timespec="seconds")


def to_relative_image_path(image_path):
    """Chemin relatif a la racine projet si possible (sinon chemin tel quel)."""
    path = os.fspath(image_path)
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        # Lecteur Windows different : on garde l'absolu.
        return path


def resolve_image_path(stored_path):
    """Inverse de to_relative_image_path : renvoie un chemin absolu utilisable."""
    path = os.fspath(stored_path)
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


class DatasetStore:
    """Acces thread-safe a l'index SQLite des echantillons."""

    def __init__(self, db_path):
        self.db_path = os.fspath(db_path)
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

    def add_sample(
        self,
        image_path,
        ocr_prediction=None,
        ocr_confidence=None,
        source="ocr",
        ocr_backend=None,
    ):
        """Insere un echantillon en attente d'etiquetage. Renvoie son id."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO samples "
                "(created_at, image_path, source, ocr_prediction, ocr_confidence, ocr_backend, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    to_relative_image_path(image_path),
                    source,
                    ocr_prediction,
                    None if ocr_confidence is None else float(ocr_confidence),
                    ocr_backend,
                    STATUS_PENDING,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid

    def get_sample(self, sample_id):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM samples WHERE id = ?", (sample_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def next_pending(self, after_id=None):
        """Prochain echantillon a etiqueter (id croissant)."""
        with self._lock:
            if after_id is None:
                row = self._conn.execute(
                    "SELECT * FROM samples WHERE status = ? ORDER BY id ASC LIMIT 1",
                    (STATUS_PENDING,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM samples WHERE status = ? AND id > ? ORDER BY id ASC LIMIT 1",
                    (STATUS_PENDING, after_id),
                ).fetchone()
        return dict(row) if row is not None else None

    def prev_pending(self, before_id):
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM samples WHERE status = ? AND id < ? ORDER BY id DESC LIMIT 1",
                (STATUS_PENDING, before_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_samples(self, statuses=None, limit=200, offset=0):
        query = "SELECT * FROM samples"
        params = []
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def iter_labeled(self):
        """Tous les echantillons porteurs d'un label exploitable (pour l'export)."""
        placeholders = ", ".join("?" for _ in LABELED_STATUSES)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM samples WHERE status IN ({placeholders}) "
                "AND label IS NOT NULL AND label != '' ORDER BY id ASC",
                LABELED_STATUSES,
            ).fetchall()
        return [dict(row) for row in rows]

    def set_label(self, sample_id, label, status):
        if status not in VALID_STATUSES:
            raise ValueError(f"Statut inconnu: {status}")
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE samples SET label = ?, status = ?, labeled_at = ? WHERE id = ?",
                (label, status, _now(), sample_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def counts_by_status(self):
        counts = {status: 0 for status in VALID_STATUSES}
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM samples GROUP BY status"
            ).fetchall()
        for row in rows:
            counts[row["status"]] = row["n"]
        counts["total"] = sum(counts[status] for status in VALID_STATUSES)
        counts["labeled"] = counts[STATUS_VALIDATED] + counts[STATUS_CORRECTED]
        return counts
