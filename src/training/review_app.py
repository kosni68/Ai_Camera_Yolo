"""Webapp de validation/correction des plaques collectees.

Affiche chaque crop avec la lecture OCR pre-remplie ; tu valides (Entree) ou tu corriges
le numero, puis tu passes au suivant. Les labels valides/corriges alimentent le dataset
d'entrainement (voir export_dataset.py).

Lancement :
    python -m src.training.review_app --host 0.0.0.0 --port 5000

Puis ouvre http://<ip-machine>:5000/ depuis ton telephone ou ton PC.
"""

import argparse
import os

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

from src.core.config import load_runtime_config
from src.ocr.plate_text import format_french_plate, match_french_plate, normalize_ocr_text
from src.training.dataset_store import (
    STATUS_CORRECTED,
    STATUS_REJECTED,
    STATUS_VALIDATED,
    DatasetStore,
    resolve_image_path,
)


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

    `backend` peut etre injecte (tests) ; sinon on charge FastPlateOcrBackend.
    """
    if backend is None:
        from src.ocr.backends.fast_plate import FastPlateOcrBackend

        backend = FastPlateOcrBackend(model_path, config_path)

    samples = store.iter_labeled()
    if limit:
        samples = samples[-int(limit):]

    rows = []
    for sample in samples:
        image = cv2.imread(resolve_image_path(sample["image_path"]))
        if image is None:
            continue
        raw = backend.read(image)
        if raw:
            match = match_french_plate(raw["raw"])
            pred = match[0] if match else normalize_ocr_text(raw["raw"])
        else:
            pred = ""
        rows.append({"id": sample["id"], "label": normalize_ocr_text(sample["label"]), "pred": pred})

    return score_predictions(rows)


def create_app(store):
    app = Flask(__name__)
    try:
        config = load_runtime_config()
        app.config["DEFAULT_MODEL"] = str(config["fast_plate_ocr_model_path"])
        app.config["DEFAULT_CONFIG"] = str(config["fast_plate_ocr_config_path"])
    except Exception:
        app.config["DEFAULT_MODEL"] = ""
        app.config["DEFAULT_CONFIG"] = ""

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
        model_path = (request.form.get("model_path") or app.config["DEFAULT_MODEL"]).strip()
        config_path = (request.form.get("config_path") or app.config["DEFAULT_CONFIG"]).strip()
        limit = (request.form.get("limit") or "").strip()

        result = None
        error = None
        if request.method == "POST":
            try:
                result = evaluate_model(
                    store, model_path, config_path, limit=int(limit) if limit else None
                )
            except Exception as exc:
                error = str(exc)

        return render_template(
            "evaluate.html",
            counts=store.counts_by_status(),
            model_path=model_path,
            config_path=config_path,
            limit=limit,
            result=result,
            error=error,
        )

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
