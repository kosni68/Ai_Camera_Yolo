"""Exporte les echantillons valides/corriges au format dataset fast-plate-ocr.

Produit, dans le dossier de sortie :
  images/<id>.jpg      les crops de plaque (copies tels quels ; fast-plate-ocr
                       redimensionne lui-meme au chargement selon plate_config)
  train.csv / val.csv  colonnes "image_path,plate_text" (chemin relatif au CSV)
  plate_config.yaml    config de preprocessing/alphabet (format FR : 7 caracteres)

Format verifie sur fast-plate-ocr 1.1.0 :
  - annotations CSV : colonnes image_path,plate_text ; image_path resolu relativement
    au dossier du CSV ; contrainte len(plate_text) <= max_plate_slots.
  - plate_config : max_plate_slots, alphabet, pad_char, img_height, img_width,
    image_color_mode (+ champs optionnels).

Lancement :
    python -m src.training.export_dataset --out data/ocr_dataset --val-ratio 0.15
"""

import argparse
import csv
import os
import random
import shutil

from src.core.config import load_runtime_config
from src.ocr.plate_text import normalize_ocr_text
from src.training.dataset_store import DatasetStore, resolve_image_path

# Plaques FR (SIV) : AA-123-AA -> 7 caracteres. L'alphabet inclut le pad '_'.
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PAD_CHAR = "_"
MAX_PLATE_SLOTS = 7
IMG_HEIGHT = 70
IMG_WIDTH = 140
IMAGE_COLOR_MODE = "grayscale"


def build_plate_config_yaml():
    return (
        "# Config de preprocessing fast-plate-ocr (genere automatiquement).\n"
        f"max_plate_slots: {MAX_PLATE_SLOTS}\n"
        f'alphabet: "{ALPHABET}"\n'
        f'pad_char: "{PAD_CHAR}"\n'
        f"img_height: {IMG_HEIGHT}\n"
        f"img_width: {IMG_WIDTH}\n"
        f"image_color_mode: {IMAGE_COLOR_MODE}\n"
    )


def collect_valid_samples(db_path):
    """Retourne [(id, chemin_image_absolu, label_normalise)] exploitables + nb ignores."""
    store = DatasetStore(db_path)
    try:
        rows = store.iter_labeled()
    finally:
        store.close()

    samples = []
    skipped = 0
    for row in rows:
        label = normalize_ocr_text(row["label"] or "")
        if not label or len(label) > MAX_PLATE_SLOTS or any(ch not in ALPHABET for ch in label):
            skipped += 1
            continue
        src_path = resolve_image_path(row["image_path"])
        if not os.path.isfile(src_path):
            skipped += 1
            continue
        samples.append((row["id"], src_path, label))
    return samples, skipped


def _write_split(out_dir, images_dir, name, items):
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "plate_text"])
        for sample_id, src_path, label in items:
            file_name = f"{sample_id}.jpg"
            shutil.copyfile(src_path, os.path.join(images_dir, file_name))
            # Chemin relatif au CSV, en forward-slash (compatible Windows et Linux).
            writer.writerow([f"images/{file_name}", label])
    return csv_path


def export(db_path, out_dir, val_ratio=0.15, seed=1234):
    samples, skipped = collect_valid_samples(db_path)
    if not samples:
        raise SystemExit(
            "Aucun echantillon etiquete exploitable. Valide d'abord des plaques "
            "via: python -m src.training.review_app"
        )

    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = int(len(samples) * val_ratio)
    if len(samples) > 1:
        n_val = max(1, n_val)
    val_items = samples[:n_val]
    train_items = samples[n_val:]

    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    train_csv = _write_split(out_dir, images_dir, "train", train_items)
    val_csv = _write_split(out_dir, images_dir, "val", val_items) if val_items else None

    config_path = os.path.join(out_dir, "plate_config.yaml")
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(build_plate_config_yaml())

    return {
        "train": len(train_items),
        "val": len(val_items),
        "skipped": skipped,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "plate_config": config_path,
        "images_dir": images_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exporte le dataset etiquete au format fast-plate-ocr."
    )
    parser.add_argument("--out", default="data/ocr_dataset", help="Dossier de sortie du dataset.")
    parser.add_argument("--db", default=None, help="Base SQLite (defaut: dataset_db_path de config.json).")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Proportion validation (0-1).")
    parser.add_argument("--seed", type=int, default=1234, help="Graine du split train/val.")
    args = parser.parse_args()

    db_path = args.db or load_runtime_config()["dataset"]["db_path"]
    result = export(db_path, args.out, val_ratio=args.val_ratio, seed=args.seed)

    print(f"[EXPORT] Train: {result['train']} | Val: {result['val']} | Ignores: {result['skipped']}")
    print(f"[EXPORT] Images : {result['images_dir']}")
    print(f"[EXPORT] CSV    : {result['train_csv']}" + (f" , {result['val_csv']}" if result["val_csv"] else ""))
    print(f"[EXPORT] Config : {result['plate_config']}")
    print("[EXPORT] Etape suivante : python -m src.training.train_ocr --dataset-dir " + args.out)


if __name__ == "__main__":
    main()
