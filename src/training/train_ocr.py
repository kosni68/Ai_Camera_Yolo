"""Assistant d'entrainement du modele OCR de plaques (fast-plate-ocr).

L'entrainement utilise Keras/TensorFlow : il est lourd et beaucoup plus rapide avec
un GPU. Deux options :

  1. En local (machine de dev avec GPU de preference) :
         pip install -r requirements/training.txt
         python -m src.training.export_dataset --out data/ocr_dataset
         python -m src.training.train_ocr --dataset-dir data/ocr_dataset --run

  2. Sur Google Colab (GPU gratuit) - recommande si pas de GPU local :
         suis le notebook officiel examples/fine_tune_workflow.ipynb de fast-plate-ocr
         en pointant sur train.csv / val.csv / plate_config.yaml exportes ici.

Par defaut ce script AFFICHE les commandes a lancer (train puis export ONNX). Avec
--run, il les execute pour toi. Le modele final est un .onnx + plate_config.yaml a
deployer (voir ocr_backend dans config.json).
"""

import argparse
import glob
import importlib.util
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_CONFIG = os.path.join(THIS_DIR, "model_config.example.yaml")

# Le CLI fast-plate-ocr s'installe comme commande "fast-plate-ocr" (tirets), mais le
# module python est "fast_plate_ocr" (underscores) : chercher la commande par son nom
# echoue selon le systeme. On l'invoque via "python -m" sur le module CLI, ce qui est
# fiable, independant du PATH, et garantit le meme interpreteur (donc le bon venv).
FAST_PLATE_OCR_CLI = [sys.executable, "-m", "fast_plate_ocr.cli.cli"]


def _fast_plate_ocr_available():
    """True si le module fast_plate_ocr est importable dans cet interpreteur."""
    return importlib.util.find_spec("fast_plate_ocr") is not None


def _require_files(dataset_dir):
    train_csv = os.path.join(dataset_dir, "train.csv")
    val_csv = os.path.join(dataset_dir, "val.csv")
    plate_config = os.path.join(dataset_dir, "plate_config.yaml")
    missing = [p for p in (train_csv, plate_config) if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "Dataset incomplet (" + ", ".join(missing) + "). Lance d'abord :\n"
            "  python -m src.training.export_dataset --out " + dataset_dir
        )
    if not os.path.isfile(val_csv):
        # fast-plate-ocr exige --val-annotations : a defaut on reutilise le train.
        val_csv = train_csv
    return train_csv, val_csv, plate_config


def build_commands(dataset_dir, model_config, output_dir, epochs, batch_size):
    train_csv, val_csv, plate_config = _require_files(dataset_dir)
    train_cmd = [
        *FAST_PLATE_OCR_CLI, "train",
        "--model-config-file", model_config,
        "--plate-config-file", plate_config,
        "--annotations", train_csv,
        "--val-annotations", val_csv,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--output-dir", output_dir,
    ]
    # Le .keras est ecrit dans <output_dir>/<timestamp>/best.keras ; l'export se fait apres.
    export_cmd = [
        *FAST_PLATE_OCR_CLI, "export",
        "--model", "<output_dir>/<timestamp>/best.keras",
        "--plate-config-file", plate_config,
        "--format", "onnx",
    ]
    return train_cmd, export_cmd, plate_config


def _print_cmd(title, cmd):
    print(f"\n# {title}")
    print("KERAS_BACKEND=tensorflow \\")
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))


def _latest_best_keras(output_dir):
    matches = glob.glob(os.path.join(output_dir, "*", "best.keras"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _run(cmd, env):
    printable = " ".join(cmd)
    print(f"\n[TRAIN] $ {printable}")
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"Commande echouee (code {completed.returncode}): {printable}")


def main():
    parser = argparse.ArgumentParser(description="Assistant d'entrainement OCR fast-plate-ocr.")
    parser.add_argument("--dataset-dir", default="data/ocr_dataset", help="Dossier produit par export_dataset.")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="Architecture (YAML).")
    parser.add_argument("--output-dir", default="models/ocr_training", help="Sortie des modeles entraines.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run", action="store_true", help="Executer (sinon, afficher les commandes).")
    args = parser.parse_args()

    train_cmd, export_cmd, plate_config = build_commands(
        args.dataset_dir, args.model_config, args.output_dir, args.epochs, args.batch_size
    )

    if not args.run:
        print("Commandes a lancer (apres: pip install -r requirements/training.txt) :")
        _print_cmd("1) Entrainement", train_cmd)
        _print_cmd("2) Export ONNX (remplace <timestamp> par le dossier cree a l'etape 1)", export_cmd)
        print(
            "\nPas de GPU ? Utilise Google Colab avec le notebook officiel "
            "examples/fine_tune_workflow.ipynb en pointant sur ce dataset."
        )
        print("Relance avec --run pour executer automatiquement en local.")
        return

    if not _fast_plate_ocr_available():
        raise SystemExit(
            "Module 'fast_plate_ocr' introuvable. Installe d'abord :\n"
            "  pip install -r requirements/training.txt"
        )

    env = dict(os.environ)
    env.setdefault("KERAS_BACKEND", "tensorflow")

    _run(train_cmd, env)

    best_keras = _latest_best_keras(args.output_dir)
    if best_keras is None:
        raise SystemExit(f"best.keras introuvable dans {args.output_dir}. Verifie les logs d'entrainement.")

    final_export = [
        *FAST_PLATE_OCR_CLI, "export",
        "--model", best_keras,
        "--plate-config-file", plate_config,
        "--format", "onnx",
    ]
    _run(final_export, env)

    onnx_path = os.path.splitext(best_keras)[0] + ".onnx"
    print("\n[TRAIN] Termine.")
    print(f"[TRAIN] Modele ONNX : {onnx_path}")
    print(f"[TRAIN] Config      : {plate_config}")
    print(
        "[TRAIN] Pour deployer : copie ces deux fichiers vers models/ puis dans config.json "
        "mets ocr_backend='fast_plate_ocr' et les chemins fast_plate_ocr_model_path / "
        "fast_plate_ocr_config_path."
    )


if __name__ == "__main__":
    main()
