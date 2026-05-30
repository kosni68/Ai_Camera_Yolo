"""Webapp de validation/correction des plaques collectees.

Affiche chaque crop avec la lecture OCR pre-remplie ; tu valides (Entree) ou tu corriges
le numero, puis tu passes au suivant. Les labels valides/corriges alimentent le dataset
d'entrainement (voir export_dataset.py).

Lancement :
    python -m src.training.review_app --host 0.0.0.0 --port 5000

Puis ouvre http://<ip-machine>:5000/ depuis ton telephone ou ton PC.
"""

import argparse
import importlib.util
import os
import time

import cv2

try:
    from flask import (
        Flask,
        abort,
        jsonify,
        redirect,
        render_template,
        request,
        send_file,
        url_for,
    )

    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

from pathlib import Path

from src.core.config import PROJECT_ROOT, load_runtime_config, update_runtime_config
from src.ocr.plate_text import format_french_plate, match_french_plate, normalize_ocr_text
from src.training.dataset_store import (
    STATUS_CORRECTED,
    STATUS_REJECTED,
    STATUS_VALIDATED,
    DatasetStore,
    resolve_image_path,
)
from src.training import model_hub
from src.training.runner import TrainingRunner


def score_predictions(rows):
    """Calcule les metriques a partir de [{id, label, pred}]. Annote chaque ligne avec 'exact'.

    Renvoie total, exact, plate_acc (%), char_acc (%) et la liste des erreurs.
    """
    total = len(rows)
    exact = 0
    char_total = 0
    char_ok = 0
    for row in rows:
        label = row["label"]
        pred = row["pred"]
        row["exact"] = pred == label and label != ""
        if row["exact"]:
            exact += 1
        char_total += max(len(label), len(pred))
        char_ok += sum(1 for i in range(min(len(label), len(pred))) if label[i] == pred[i])

    return {
        "total": total,
        "exact": exact,
        "plate_acc": (exact / total * 100.0) if total else 0.0,
        "char_acc": (char_ok / char_total * 100.0) if char_total else 0.0,
        "errors": [row for row in rows if not row["exact"]],
    }


def evaluate_model(store, model_path, config_path, limit=None, backend=None):
    """Fait tourner le modele sur les plaques etiquetees et renvoie les metriques.

    En plus de la precision (plaque/caractere), on mesure des indicateurs de
    performance : temps de chargement du modele, temps d'inference moyen par plaque,
    debit (plaques/seconde) et taille du fichier .onnx. Ces chiffres permettent de
    comparer vitesse *et* precision entre deux modeles avant de deployer.

    `backend` peut etre injecte (tests) ; sinon on charge FastPlateOcrBackend.
    """
    load_ms = None
    if backend is None:
        from src.ocr.backends.fast_plate import FastPlateOcrBackend

        load_start = time.perf_counter()
        backend = FastPlateOcrBackend(model_path, config_path)
        load_ms = (time.perf_counter() - load_start) * 1000.0

    samples = store.iter_labeled()
    if limit:
        samples = samples[-int(limit):]

    rows = []
    inference_ms_total = 0.0
    inference_count = 0
    for sample in samples:
        image = cv2.imread(resolve_image_path(sample["image_path"]))
        if image is None:
            continue
        infer_start = time.perf_counter()
        raw = backend.read(image)
        inference_ms_total += (time.perf_counter() - infer_start) * 1000.0
        inference_count += 1
        if raw:
            match = match_french_plate(raw["raw"])
            pred = match[0] if match else normalize_ocr_text(raw["raw"])
        else:
            pred = ""
        rows.append({"id": sample["id"], "label": normalize_ocr_text(sample["label"]), "pred": pred})

    metrics = score_predictions(rows)
    avg_ms = (inference_ms_total / inference_count) if inference_count else 0.0
    metrics["perf"] = {
        "load_ms": load_ms,
        "avg_ms": avg_ms,
        "total_ms": inference_ms_total,
        "fps": (1000.0 / avg_ms) if avg_ms else 0.0,
        "samples_timed": inference_count,
        "model_size_mb": _file_size_mb(model_path),
    }
    return metrics


def _file_size_mb(path):
    """Taille du fichier en Mo, ou None si introuvable."""
    try:
        return os.path.getsize(_resolve_under_root(path)) / (1024.0 * 1024.0)
    except OSError:
        return None


def _resolve_under_root(path):
    """Chemin absolu : relatif => resolu depuis PROJECT_ROOT (cwd de la webapp)."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _active_ocr_paths():
    """Chemins absolus du modele OCR actuellement configure (vide si indispo)."""
    try:
        config = load_runtime_config()
        return str(config["fast_plate_ocr_model_path"]), str(config["fast_plate_ocr_config_path"])
    except Exception:
        return "", ""


def deploy_model(model_path, config_path):
    """Active ce modele dans config.json (backend fast_plate_ocr) sans copie manuelle.

    Remplace l'etape manuelle de fin de train_ocr.py : on pointe config.json sur le
    modele entraine la ou il est (dossier d'entrainement), apres avoir verifie que les
    fichiers existent. Renvoie les chemins relatifs ecrits dans la config.
    """
    model_abs = _resolve_under_root(model_path)
    config_abs = _resolve_under_root(config_path)
    if not model_abs.is_file():
        raise RuntimeError(f"Modele introuvable : {model_abs}")
    if not config_abs.is_file():
        raise RuntimeError(f"Config du modele introuvable : {config_abs}")

    def _rel(path):
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)

    model_rel = _rel(model_abs)
    config_rel = _rel(config_abs)
    update_runtime_config(
        {
            "ocr_backend": "fast_plate_ocr",
            "fast_plate_ocr_model_path": model_rel,
            "fast_plate_ocr_config_path": config_rel,
        }
    )
    return {"model": model_rel, "config": config_rel}


def create_app(store):
    app = Flask(__name__)
    try:
        config = load_runtime_config()
        app.config["DEFAULT_MODEL"] = str(config["fast_plate_ocr_model_path"])
        app.config["DEFAULT_CONFIG"] = str(config["fast_plate_ocr_config_path"])
    except Exception:
        app.config["DEFAULT_MODEL"] = ""
        app.config["DEFAULT_CONFIG"] = ""

    runner = TrainingRunner(
        dataset_dir=PROJECT_ROOT / "data" / "ocr_dataset",
        output_dir=PROJECT_ROOT / "models" / "ocr_training",
        log_path=PROJECT_ROOT / "data" / "ocr_training.log",
    )
    app.config["RUNNER"] = runner

    @app.route("/")
    def index():
        sample = store.next_pending()
        if sample is not None:
            return redirect(url_for("review", sample_id=sample["id"]))
        return render_template(
            "review.html",
            sample=None,
            counts=store.counts_by_status(),
            prefill="",
            nxt=None,
            prv=None,
        )

    @app.route("/review/<int:sample_id>")
    def review(sample_id):
        sample = store.get_sample(sample_id)
        if sample is None:
            abort(404)
        prefill = format_french_plate(sample.get("ocr_prediction") or "")
        return render_template(
            "review.html",
            sample=sample,
            counts=store.counts_by_status(),
            prefill=prefill,
            nxt=store.next_pending(after_id=sample_id),
            prv=store.prev_pending(before_id=sample_id),
        )

    @app.route("/label/<int:sample_id>", methods=["POST"])
    def label(sample_id):
        sample = store.get_sample(sample_id)
        if sample is None:
            abort(404)

        action = request.form.get("action", "save")
        if action == "reject":
            store.set_label(sample_id, "", STATUS_REJECTED)
        elif action == "skip":
            pass  # on laisse le statut 'pending', on avance juste
        else:
            normalized = normalize_ocr_text(request.form.get("label", ""))
            if not normalized:
                store.set_label(sample_id, "", STATUS_REJECTED)
            else:
                predicted = normalize_ocr_text(sample.get("ocr_prediction") or "")
                status = STATUS_VALIDATED if normalized == predicted else STATUS_CORRECTED
                store.set_label(sample_id, normalized, status)

        nxt = store.next_pending(after_id=sample_id) or store.next_pending()
        if nxt is None:
            return redirect(url_for("index"))
        return redirect(url_for("review", sample_id=nxt["id"]))

    @app.route("/image/<int:sample_id>")
    def image(sample_id):
        sample = store.get_sample(sample_id)
        if sample is None:
            abort(404)
        path = resolve_image_path(sample["image_path"])
        if not os.path.isfile(path):
            abort(404)
        return send_file(path)

    @app.route("/api/stats")
    def stats():
        return jsonify(store.counts_by_status())

    @app.route("/evaluate", methods=["GET", "POST"])
    def evaluate():
        active_model, active_config = _active_ocr_paths()
        model_path = (request.form.get("model_path") or active_model or app.config["DEFAULT_MODEL"]).strip()
        config_path = (request.form.get("config_path") or active_config or app.config["DEFAULT_CONFIG"]).strip()
        limit = (request.form.get("limit") or "").strip()
        action = request.form.get("action", "evaluate")

        result = None
        error = request.args.get("dl_error")
        deployed = None
        downloaded = request.args.get("downloaded")
        if request.method == "POST":
            try:
                if action == "deploy":
                    deployed = deploy_model(model_path, config_path)
                    active_model, active_config = _active_ocr_paths()
                else:
                    result = evaluate_model(
                        store, model_path, config_path, limit=int(limit) if limit else None
                    )
            except Exception as exc:
                error = str(exc)

        models = runner.list_models()
        active_norm = os.path.normcase(os.path.abspath(active_model)) if active_model else ""
        for model in models:
            model["is_active"] = (
                os.path.normcase(os.path.abspath(model["model_path"])) == active_norm
            )
            if model["kind"] == "origin":
                model["label"] = model_hub.friendly_label(model["model_path"]) or model["name"]
            else:
                model["label"] = model["name"]

        return render_template(
            "evaluate.html",
            counts=store.counts_by_status(),
            model_path=model_path,
            config_path=config_path,
            limit=limit,
            result=result,
            error=error,
            models=models,
            deployed=deployed,
            hub_catalog=model_hub.catalog(),
            downloaded=downloaded,
        )

    @app.route("/models/download", methods=["POST"])
    def download_model_route():
        """Telecharge un modele pre-entraine du hub dans models/, puis revient a /evaluate."""
        model_id = (request.form.get("model_id") or "").strip()
        try:
            model_hub.download(model_id)
            return redirect(url_for("evaluate", downloaded=model_id))
        except Exception as exc:
            return redirect(url_for("evaluate", dl_error=str(exc)))

    @app.route("/train")
    def train_page():
        return render_template(
            "train.html",
            counts=store.counts_by_status(),
            state=runner.state(),
            log=runner.read_log(),
            tool_available=importlib.util.find_spec("fast_plate_ocr") is not None,
            output_dir=str(runner.output_dir),
        )

    @app.route("/train/start", methods=["POST"])
    def train_start():
        epochs = request.form.get("epochs") or 150
        batch_size = request.form.get("batch_size") or 64
        runner.start(epochs=epochs, batch_size=batch_size)
        return redirect(url_for("train_page"))

    @app.route("/train/status")
    def train_status():
        state = runner.state()
        state["log"] = runner.read_log()
        return jsonify(state)

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Webapp de validation/correction des plaques collectees."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'ecoute (0.0.0.0 = accessible LAN).")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--db", default=None, help="Chemin de la base (defaut: dataset_db_path de config.json).")
    args = parser.parse_args()

    if not _FLASK_AVAILABLE:
        raise SystemExit(
            "Flask n'est pas installe. Lance: pip install -r requirements/training.txt"
        )

    db_path = args.db
    if db_path is None:
        db_path = load_runtime_config()["dataset"]["db_path"]

    store = DatasetStore(db_path)
    app = create_app(store)
    print(f"[REVIEW] Base: {db_path}")
    print(f"[REVIEW] Ouvre http://{args.host}:{args.port}/ dans ton navigateur")
    try:
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        store.close()


if __name__ == "__main__":
    main()
