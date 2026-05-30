"""Importe des crops de plaque deja sauvegardes dans la base de validation.

Utile pour recuperer les images deja capturees par save_plates_enabled
(dossier data/plates/) sans avoir a tout recollecter. Chaque image devient un
echantillon "pending" a valider dans la webapp. Si le nom de fichier contient la
plaque lue (format "<horodatage>__AB123CD.jpg"), elle pre-remplit la prediction.

Lancement :
    python -m src.training.import_plates --folder data/plates
"""

import argparse
import glob
import os

from src.core.config import load_runtime_config
from src.training.dataset_store import DatasetStore, to_relative_image_path

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def extract_prediction(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    if "__" in stem:
        return stem.split("__", 1)[1]
    return ""


def import_folder(db_path, folder, source="imported"):
    if not os.path.isdir(folder):
        raise SystemExit(f"Dossier introuvable: {folder}")

    store = DatasetStore(db_path)
    try:
        existing = store.existing_image_paths()
        images = []
        for pattern in IMAGE_EXTENSIONS:
            images.extend(glob.glob(os.path.join(folder, "**", pattern), recursive=True))

        added = 0
        skipped = 0
        for image_path in sorted(images):
            if to_relative_image_path(image_path) in existing:
                skipped += 1
                continue
            prediction = extract_prediction(image_path)
            store.add_sample(image_path, ocr_prediction=prediction or None, source=source)
            added += 1
        counts = store.counts_by_status()
    finally:
        store.close()

    return added, skipped, len(images), counts


def main():
    parser = argparse.ArgumentParser(
        description="Importe des crops de plaque existants dans la base de validation."
    )
    parser.add_argument("--folder", default="data/plates", help="Dossier d'images a importer.")
    parser.add_argument("--db", default=None, help="Base SQLite (defaut: dataset_db_path de config.json).")
    parser.add_argument("--source", default="imported", help="Etiquette de source des echantillons.")
    args = parser.parse_args()

    db_path = args.db or load_runtime_config()["dataset"]["db_path"]
    added, skipped, total, counts = import_folder(db_path, args.folder, source=args.source)

    print(f"[IMPORT] Images trouvees: {total} | ajoutees: {added} | deja presentes: {skipped}")
    print(f"[IMPORT] Base: {db_path} | en attente: {counts['pending']}")
    print("[IMPORT] Valide-les avec : python -m src.training.review_app --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()
