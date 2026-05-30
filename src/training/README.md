# Boucle d'apprentissage OCR (entraîner ton propre lecteur de plaques)

Ce module te permet d'**entraîner un modèle de lecture de plaques sur TES plaques et TA
caméra**, en validant/corrigeant toi-même les lectures. Le cycle :

```
1. COLLECTE     le pipeline enregistre chaque crop de plaque + la lecture OCR
2. VALIDATION   tu valides (✓) ou corriges le numéro depuis une page web
3. EXPORT       les plaques étiquetées deviennent un dataset (image → "AB123CD")
4. ENTRAÎNEMENT un modèle fast-plate-ocr apprend sur ce dataset
5. DÉPLOIEMENT  le pipeline lit les plaques avec TON modèle
```

Plus tu valides/corriges, meilleur devient le modèle. Idéal pour une caméra fixe : peu de
plaques, conditions de prise de vue constantes.

---

## Étape 1 — Collecter des exemples

Dans [config/config.json](../../config/config.json), active la collecte :

```json
"dataset_collection_enabled": true
```

Puis lance le pipeline normalement :

```bash
python -m src.main
```

À chaque plaque traitée, un crop est enregistré dans `data/dataset/images/` et indexé dans
`data/dataset/samples.db`. Deux garde-fous évitent de saturer le disque (`dataset_dedup_window_sec`,
`dataset_max_per_minute`). Laisse tourner quelques jours pour accumuler de la variété (jour/nuit,
angles, météo). **Vise au moins ~200 plaques étiquetées** avant d'entraîner ; plus = mieux.

> Tu as déjà des photos dans `data/plates/` (option `save_plates_enabled`) ? Importe-les dans la
> base sans tout recollecter :
> ```bash
> python -m src.training.import_plates --folder data/plates
> ```

## Étape 2 — Valider / corriger (webapp)

Installe l'outillage (une fois) puis lance la webapp :

```bash
pip install -r requirements/training.txt
python -m src.training.review_app --host 0.0.0.0 --port 5000
```

Ouvre `http://<ip-de-la-machine>:5000/` depuis ton PC ou ton téléphone. Pour chaque crop :

- le champ est **pré-rempli** avec la lecture OCR ;
- **Entrée** = valider tel quel · tape le bon numéro puis Entrée = corriger ;
- **r** = rejeter (image illisible) · **p** = passer · **←/→** = naviguer.

Les numéros sont stockés normalisés (`AB123CD`, sans tirets), au format attendu par l'entraînement.

## Étape 3 — Exporter le dataset

```bash
python -m src.training.export_dataset --out data/ocr_dataset --val-ratio 0.15
```

Produit `train.csv`, `val.csv`, `plate_config.yaml` et les images dans `data/ocr_dataset/`.
Les plaques de plus de 7 caractères ou aux fichiers manquants sont ignorées (compte affiché).

## Étape 4 — Entraîner

L'entraînement utilise Keras/TensorFlow : **lent sur CPU, rapide sur GPU**.

**Le plus simple : depuis la webapp.** Clique sur **🚀 Entrainer** (page `/train`), choisis le nombre
d'epochs, puis lance. La page enchaîne export -> entraînement -> export ONNX **en arrière-plan** et
affiche les **logs en direct** + le statut. (À faire sur la machine de dev, pas le Raspberry Pi.)

En ligne de commande, voir les commandes à lancer :

```bash
python -m src.training.train_ocr --dataset-dir data/ocr_dataset
```

Pour exécuter en local (après `pip install -r requirements/training.txt`) :

```bash
python -m src.training.train_ocr --dataset-dir data/ocr_dataset --run
```

Cela lance l'entraînement (`fast_plate_ocr train`) puis l'export ONNX (`fast_plate_ocr export`),
et produit un `best.onnx` + `plate_config.yaml`.

**Pas de GPU ?** Utilise **Google Colab** (GPU gratuit) avec le notebook officiel
`examples/fine_tune_workflow.ipynb` de fast-plate-ocr, en pointant sur ton `train.csv` / `val.csv` /
`plate_config.yaml`. Le **fine-tuning** d'un modèle pré-entraîné (européen) donne de bons résultats
avec peu de données — bien mieux que l'architecture d'exemple [model_config.example.yaml](model_config.example.yaml)
entraînée de zéro.

## Étape 5 — Déployer ton modèle

Copie `best.onnx` et `plate_config.yaml` dans `models/`, puis dans [config/config.json](../../config/config.json) :

```json
"ocr_backend": "fast_plate_ocr",
"fast_plate_ocr_model_path": "models/best.onnx",
"fast_plate_ocr_config_path": "models/plate_config.yaml"
```

Installe l'inférence : `pip install "fast-plate-ocr[onnx]"` (voir la note numpy dans
[requirements/optional.txt](../../requirements/optional.txt)). Relance `python -m src.main` : le
worker lit désormais avec ton modèle (`[OCR] Backend fast-plate-ocr actif`). Si le modèle est
absent/illisible, il bascule automatiquement sur EasyOCR/Tesseract.

## Mesurer si le modèle s'améliore

Dans la webapp, clique sur **🧪 Tester le modèle** (page `/evaluate`). Indique le chemin du `.onnx`
et de son `plate_config.yaml`, puis lance l'évaluation : le modèle est exécuté sur tes plaques
**déjà validées** (la vérité terrain) et la page affiche le **% de plaques exactes**, la précision
caractère et la liste des erreurs (vignette + attendu vs lu). Relance après chaque entraînement pour
comparer. Astuce : note le score à chaque itération pour suivre la progression.

---

## Bonus — Ouverture fiable du portail (fuzzy-matching)

Indépendamment de l'entraînement, `registered_plate_fuzzy_distance` (défaut `1`) tolère N
caractères d'écart entre la lecture et une plaque de `config/registered_plates.json`. Ainsi `AB-I23-CD`
(I lu au lieu de 1) ouvre quand même le portail. Mets `0` pour exiger une correspondance exacte
(plus sûr, recommandé si des plaques voisines se ressemblent).

## Référence config

| Clé | Défaut | Rôle |
|-----|--------|------|
| `dataset_collection_enabled` | `false` | Active la collecte d'échantillons |
| `dataset_db_path` | `data/dataset/samples.db` | Base SQLite de l'index |
| `dataset_image_root` | `data/dataset/images` | Dossier des crops collectés |
| `dataset_dedup_window_sec` | `8.0` | Anti-doublon : même lecture rapprochée ignorée |
| `dataset_max_per_minute` | `30` | Plafond d'échantillons/minute |
| `ocr_backend` | `auto` | `auto` (EasyOCR/Tesseract) ou `fast_plate_ocr` |
| `fast_plate_ocr_model_path` | `models/fast_plate_ocr.onnx` | Modèle ONNX entraîné |
| `fast_plate_ocr_config_path` | `models/fast_plate_ocr_config.yaml` | Config du modèle |
| `registered_plate_fuzzy_distance` | `1` | Tolérance liste blanche (0 = exact) |

## Fichiers

- `dataset_store.py` — index SQLite (thread-safe)
- `collector.py` — sauvegarde crop + insertion (branché au worker OCR)
- `import_plates.py` — importe des crops déjà sauvegardés (`data/plates/`) dans la base
- `review_app.py` + `templates/review.html` — webapp de validation
- `templates/evaluate.html` — page de test/évaluation du modèle (`/evaluate`)
- `runner.py` + `templates/train.html` — lancement d'entraînement en arrière-plan (`/train`)
- `export_dataset.py` — export au format fast-plate-ocr
- `train_ocr.py` + `model_config.example.yaml` — assistant d'entraînement (CLI)
- backend de déploiement : [../ocr/backends/fast_plate.py](../ocr/backends/fast_plate.py)
