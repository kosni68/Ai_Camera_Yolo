"""Telechargement de modeles OCR pre-entraines (hub fast-plate-ocr).

fast-plate-ocr publie plusieurs modeles ONNX prets a l'emploi (CCT, MobileViT) sur
GitHub. On les telecharge directement dans `models/` : le scanner `list_models()` du
runner les detecte alors comme modeles "d'origine" et ils apparaissent dans la liste
de l'interface web (page /evaluate), sans copie manuelle.

Chaque modele vient avec son `*_config.yaml` / `*_plate_config.yaml` compagnon, place
a cote du .onnx, ce qui suffit a `_find_companion_config()` pour faire le lien.

Usage CLI :
    python -m src.training.model_hub            # set recommande
    python -m src.training.model_hub --all      # tous les modeles
    python -m src.training.model_hub cct-xs-v2-global-model
"""

import os

from src.core.config import PROJECT_ROOT

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Catalogue presente dans l'UI. Chaque entree : id hub -> metadonnees lisibles.
# `recommended` = telecharge par defaut (modeles pertinents pour des plaques FR/EU).
HUB_MODELS = {
    "european-plates-mobile-vit-v2-model": {
        "label": "Europeen (MobileViT v2)",
        "description": "Plaques europeennes, le plus precis pour la France. ~MobileViT v2.",
        "recommended": True,
    },
    "global-plates-mobile-vit-v2-model": {
        "label": "Global (MobileViT v2)",
        "description": "Plaques du monde entier (Europe incluse). Precis mais plus lourd.",
        "recommended": True,
    },
    "cct-xs-v2-global-model": {
        "label": "Global CCT-XS v2 (rapide)",
        "description": "Petit transformeur CCT, tres rapide. Bon compromis vitesse/precision.",
        "recommended": True,
    },
    "cct-s-v2-global-model": {
        "label": "Global CCT-S v2 (precis)",
        "description": "CCT plus grand, plus precis que XS, un peu plus lent.",
        "recommended": False,
    },
    "cct-xs-v1-global-model": {
        "label": "Global CCT-XS v1",
        "description": "Version 1 du petit CCT global.",
        "recommended": False,
    },
    "cct-s-v1-global-model": {
        "label": "Global CCT-S v1",
        "description": "Version 1 du CCT-S global.",
        "recommended": False,
    },
    "argentinian-plates-cnn-model": {
        "label": "Argentine (CNN)",
        "description": "CNN dedie aux plaques argentines. Reference rapide.",
        "recommended": False,
    },
}

RECOMMENDED = tuple(name for name, meta in HUB_MODELS.items() if meta["recommended"])


def _hub_download(model_name, save_directory, force=False):
    """Appelle le telechargeur de fast-plate-ocr. Importe a la demande (dep optionnelle)."""
    import pathlib

    from fast_plate_ocr.inference.hub import download_model

    return download_model(
        model_name, save_directory=pathlib.Path(save_directory), force_download=force
    )


def installed_paths(model_name):
    """(model_path, config_path) si le modele est deja dans models/, sinon None.

    On ne re-telecharge pas : on derive juste les noms de fichiers attendus a partir
    des URLs du hub et on verifie leur presence sur disque.
    """
    try:
        from fast_plate_ocr.inference.hub import AVAILABLE_ONNX_MODELS
    except ImportError:
        return None

    entry = AVAILABLE_ONNX_MODELS.get(model_name)
    if entry is None:
        return None

    model_url, config_url = entry
    model_path = os.path.join(MODELS_DIR, model_url.split("/")[-1])
    config_path = os.path.join(MODELS_DIR, config_url.split("/")[-1])
    if os.path.isfile(model_path) and os.path.isfile(config_path):
        return model_path, config_path
    return None


def friendly_label(onnx_path):
    """Label lisible pour un .onnx du hub (ex: 'Europeen (MobileViT v2)'), sinon None.

    Fait le lien entre le nom de fichier sur disque (ex: european_mobile_vit_v2_ocr.onnx)
    et l'id hub correspondant, via les URLs de AVAILABLE_ONNX_MODELS.
    """
    try:
        from fast_plate_ocr.inference.hub import AVAILABLE_ONNX_MODELS
    except ImportError:
        return None

    filename = os.path.basename(os.fspath(onnx_path))
    for model_id, (model_url, _config_url) in AVAILABLE_ONNX_MODELS.items():
        if model_url.split("/")[-1] == filename:
            meta = HUB_MODELS.get(model_id)
            return meta["label"] if meta else model_id
    return None


def catalog():
    """Liste enrichie pour l'UI : id, label, description, recommended, installed."""
    rows = []
    for name, meta in HUB_MODELS.items():
        rows.append(
            {
                "id": name,
                "label": meta["label"],
                "description": meta["description"],
                "recommended": meta["recommended"],
                "installed": installed_paths(name) is not None,
            }
        )
    return rows


def download(model_name, force=False):
    """Telecharge un modele du hub dans models/. Renvoie (model_path, config_path)."""
    if model_name not in HUB_MODELS:
        available = ", ".join(HUB_MODELS)
        raise ValueError(f"Modele hub inconnu : {model_name}. Choisis parmi : {available}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path, config_path = _hub_download(model_name, MODELS_DIR, force=force)
    return os.fspath(model_path), os.fspath(config_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Telecharge des modeles OCR pre-entraines.")
    parser.add_argument(
        "models",
        nargs="*",
        help="ids hub a telecharger (defaut: set recommande). Voir --list.",
    )
    parser.add_argument("--all", action="store_true", help="Telecharge tous les modeles du catalogue.")
    parser.add_argument("--force", action="store_true", help="Re-telecharge meme si deja present.")
    parser.add_argument("--list", action="store_true", help="Affiche le catalogue et quitte.")
    args = parser.parse_args()

    if args.list:
        for row in catalog():
            flag = "[installe]" if row["installed"] else "          "
            star = "*" if row["recommended"] else " "
            print(f"{flag} {star} {row['id']:40s} {row['label']}")
        return

    if args.all:
        targets = list(HUB_MODELS)
    elif args.models:
        targets = args.models
    else:
        targets = list(RECOMMENDED)

    for name in targets:
        print(f"== {name} ==")
        try:
            model_path, config_path = download(name, force=args.force)
            print(f"   OK  {os.path.relpath(model_path, PROJECT_ROOT)}")
            print(f"       {os.path.relpath(config_path, PROJECT_ROOT)}")
        except Exception as exc:
            print(f"   ECHEC : {exc}")


if __name__ == "__main__":
    main()
