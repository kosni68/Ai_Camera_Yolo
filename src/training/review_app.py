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
from src.ocr.plate_text import format_french_plate, normalize_ocr_text
from src.training.dataset_store import (
    STATUS_CORRECTED,
    STATUS_REJECTED,
    STATUS_VALIDATED,
    DatasetStore,
    resolve_image_path,
)


def create_app(store):
    app = Flask(__name__)

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
