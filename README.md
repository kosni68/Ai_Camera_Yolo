
# 📹 Streaming RTSP depuis une caméra Raspberry Pi avec MediaMTX

Ce guide explique comment configurer un flux vidéo **RTSP** et **WebRTC** depuis une caméra Raspberry Pi (libcamera), en utilisant :

- **MediaMTX** (anciennement rtsp-simple-server)
- **libcamera-vid**
- **FFmpeg**
- Un **service systemd** pour lancer automatiquement le flux au démarrage

---

## 🚀 1. Installer MediaMTX via Docker

Lancer MediaMTX en mode host pour exposer les ports RTSP / WebRTC :

```bash
sudo docker pull bluenviron/mediamtx:latest
```

```bash
sudo docker run -d \
  --network host \
  --device /dev/video0 \
  --restart unless-stopped \
  bluenviron/mediamtx:latest

```

Cela démarre un serveur RTSP sur :

- **RTSP** → `rtsp://<IP>:8554`
- **WebRTC** → `http://<IP>:8889`

---

## 🎥 2. Lancer un flux RTSP via libcamera et FFmpeg

Commande manuelle pour tester le flux :

```bash
libcamera-vid -t 0 --width 1280 --height 720 --rotation 180 --codec yuv420 --inline -n -o - | ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 1280x720 -i - -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:8554/mystream
```

```bash
libcamera-vid -t 0 --width 4608 --height 2592 --rotation 180 --codec yuv420 --inline -n -o - | ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 4608x2592 -i - -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:8554/mystream
```

```bash
libcamera-vid -t 0 --width 4608 --height 2592 --framerate 14 --rotation 180 \
  --codec yuv420 --inline -n -o - | \
ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 4608x2592 -framerate 14 -i - \
  -an -c:v libx264 -preset ultrafast -tune zerolatency \
  -g 28 -keyint_min 28 -sc_threshold 0 \
  -crf 28 \
  -f rtsp -rtsp_transport tcp -muxdelay 0 -muxpreload 0 \
  rtsp://localhost:8554/mystream
```
Si tu veux meilleure qualité : baisse -crf (ex: 26).
Si tu veux moins de CPU : monte -crf (ex: 30).


```bash
libcamera-vid -t 0 --width 4608 --height 2592 --framerate 14 --rotation 180 \
  --codec yuv420 --inline -n -o - | \
ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 4608x2592 -framerate 14 -i - \
  -an -c:v libx264 -preset ultrafast -tune zerolatency \
  -g 28 -keyint_min 28 -sc_threshold 0 \
  -b:v 12M -maxrate 12M -bufsize 24M \
  -f rtsp -rtsp_transport tcp -muxdelay 0 -muxpreload 0 \
  rtsp://localhost:8554/mystream
```

```bash
libcamera-vid -t 0 --mode 4608:2592:10 --width 4096 --height 2304 --framerate 14 --rotation 180 \
  --codec yuv420 --inline -n -o - | \
ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 4096x2304 -framerate 14 -i - \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -x264-params "keyint=14:min-keyint=14:scenecut=0:repeat-headers=1" \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/mystream
```
Le flux sera disponible à ces adresses :

### ▶️ Lecture RTSP
```
rtsp://<IP_RPI>:8554/mystream
```

### 🌐 Lecture WebRTC
```
http://<IP_RPI>:8889/mystream
```

---

## 🔧 3. Créer un service systemd

Créer le fichier :

```bash
sudo nano /etc/systemd/system/rpi-rtsp.service
```

Contenu :

```ini
[Unit]
Description=Camera RTSP stream (libcamera-vid -> ffmpeg -> MediaMTX)
After=network.target mediamtx.service
Wants=network.target

[Service]
Type=simple
ExecStart=/bin/bash -lc 'libcamera-vid -t 0 --mode 4608:2592:10 --width 4096 --height 2304 --framerate 14 --codec yuv420 --rotation 180 --inline -n -o - | ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 4096x2304 -framerate 14 -i - -an -c:v libx264 -preset ultrafast -tune zerolatency -x264-params "keyint=14:min-keyint=14:scenecut=0:repeat-headers=1" -f rtsp -rtsp_transport tcp rtsp://localhost:8554/mystream'
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target

```

---

## ▶️ 4. Activer et démarrer le service

```bash
sudo systemctl daemon-reload
sudo systemctl enable rpi-rtsp.service
sudo systemctl start rpi-rtsp.service
```

---

## 🧪 5. Tester le flux

RTSP :
```
rtsp://<IP_RPI>:8554/mystream
```

WebRTC :
```
http://<IP_RPI>:8889/mystream
```

---

## ✔️ Résumé

| Fonction | Commande / URL |
|---------|----------------|
| Stream RTSP | `rtsp://<IP>:8554/mystream` |
| Stream WebRTC | `http://<IP>:8889/mystream` |
| Démarrer service | `sudo systemctl start rpi-rtsp` |
| Activer au boot | `sudo systemctl enable rpi-rtsp` |
---

## Structure du projet

```
Ai_Camera_Yolo/
├── Dockerfile               # Image Docker pour le pipeline de detection
├── docker-compose.yml       # Compose : monte config/ et data/, reseau host
├── .dockerignore
├── config/
│   └── config.json          # Parametres runtime (RTSP, OCR, ROI, FPS...)
├── models/                  # Modeles YOLO (non versionnes)
│   ├── license_plate_detector.pt      # Detecteur principal (vehicules)
│   └── anpr_best.pt                   # Second detecteur (affinage plaque)
├── data/                    # Logs et captures (non versionnes)
│   ├── detected_plates.txt
│   ├── fps_stats.txt
│   ├── history/
│   ├── detections/
│   └── benchmarks/
├── requirements/
│   ├── base.txt             # Dependances principales
│   └── optional.txt         # EasyOCR (optionnel)
├── scripts/
│   ├── setup_env.ps1        # Setup Windows
│   └── setup_env.sh         # Setup Linux/Ubuntu
└── src/
    ├── main.py              # Entry point : pipeline de detection
    ├── benchmark.py         # Entry point : benchmark RTSP
    ├── core/
    │   ├── capture.py       # FrameGrabber RTSP
    │   ├── config.py        # Chargement et validation de config.json
    │   └── motion.py        # Detecteur de mouvement (gate leger avant YOLO)
    ├── detection/
    │   ├── yolo.py          # Chargement YOLO, extraction detections, ROI, sauvegarde
    │   └── plate.py         # Second modele : recadrage plaque
    ├── ocr/
    │   ├── worker.py        # PlateOcrWorker + backends EasyOCR/Tesseract
    │   └── plate_text.py    # Matching format FR, preprocessing image
    └── utils/
        ├── drawing.py       # Overlay FPS
        └── logging.py       # Ecriture atomique, historique journalier
```

## Windows setup

Depuis la racine du repo :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
python -m src.main
```

Notes utiles :

- Le script cree ou reutilise `.venv` a la racine du repo.
- Python 3.12 est requis.
- Tesseract est installe automatiquement via `winget` s'il n'est pas deja present.
- Pour installer les dependances optionnelles EasyOCR : `python -m pip install -r .\requirements\optional.txt`
- Si `tesseract` n'est pas reconnu juste apres l'installation, ouvre un nouveau terminal PowerShell.
- Les reglages runtime sont charges depuis `config/config.json`.
- `video_display_enabled` active ou coupe toutes les fenetres OpenCV.
- `overlay_enabled` coupe les overlays du flux principal.
- `ocr_enabled` active ou coupe tout le pipeline OCR.
- `ocr_fast_mode_enabled` active un profil OCR plus leger.
- `ocr_submit_interval_sec` espace les jobs OCR pour eviter de retraiter trop souvent la meme scene.
- `ocr_same_crop_retry_sec` retarde les retries quand le crop plaque change tres peu.
- `secondary_plate_detector_enabled` active ou coupe le second modele YOLO qui affine le crop plaque avant OCR.
- `secondary_plate_detector_model_path` chemin vers le modele du second detecteur (defaut : `models/anpr_best.pt`).
- `save_detections_enabled` active la sauvegarde des crops de detection en JPEG dans `detection_save_root`.
- `detection_save_min_confidence` seuil minimum de confiance pour declencher la sauvegarde (ex : `0.5`).
- `detection_save_root` dossier racine pour les crops sauvegardes (defaut : `data/detections`).
- `motion_detection_enabled` active le gate de mouvement : YOLO n'est lance que si du mouvement est detecte.
- `motion_resize_width` largeur cible pour le redimensionnement avant calcul de difference (ex : `320`).
- `motion_diff_threshold` seuil pixel pour considerer un pixel comme "different" (ex : `25`).
- `motion_min_area_ratio` ratio minimal de pixels en mouvement pour valider une detection (ex : `0.015`).
- `motion_keepalive_sec` duree pendant laquelle YOLO reste actif apres la fin du mouvement (ex : `1.0`).
- `motion_force_detector_interval_sec` force une inference YOLO meme sans mouvement apres ce delai en secondes (ex : `10.0`).
- `fps_limit` plafonne la boucle principale.
- `detector_fps_limit` cadence les inferences YOLO pour reduire la charge CPU.
- `roi_enabled`, `roi_x`, `roi_y`, `roi_width` et `roi_height` limitent l'analyse a une zone normalisee.

## Ubuntu setup

Depuis la racine du repo :

```bash
chmod +x ./scripts/setup_env.sh
./scripts/setup_env.sh
source ./.venv/bin/activate
python -m src.main
```

Notes utiles :

- Le script cree ou reutilise `.venv` a la racine du repo.
- Python 3.12 est requis.
- Le script installe automatiquement `tesseract-ocr` et les bibliotheques systeme OpenCV via `apt-get`.
- Sur une machine sans interface graphique, mets `video_display_enabled` a `false` dans `config/config.json`.
- Si `video_display_enabled` reste a `true` sur un Linux sans `DISPLAY` ni `WAYLAND_DISPLAY`, le script desactive automatiquement les fenetres OpenCV et les overlays.
- Pour installer les dependances optionnelles EasyOCR : `python -m pip install -r ./requirements/optional.txt`

## Docker

Le `Dockerfile` produit une image `python:3.12-slim` avec Tesseract FR et les dependances OpenCV preinstallees. Le `docker-compose.yml` monte `config/` et `data/` en volume et utilise `network_mode: host` pour atteindre le flux RTSP local.

Construire et lancer :

```bash
docker compose up --build -d
```

Suivre les logs :

```bash
docker compose logs -f camera
```

Notes :

- Le conteneur ne gere pas de fenetre graphique : mets `video_display_enabled` a `false` dans `config/config.json`.
- Les plaques detectees et les stats FPS sont ecrites dans `data/` sur l'hote via le volume.
- Pour passer des modeles custom, place-les dans `models/` avant le build (ils sont copies dans l'image via `COPY models/ models/`).
- EasyOCR n'est pas inclus dans l'image de base ; si besoin, ajoute `requirements/optional.txt` dans le `Dockerfile`.

---

## Benchmark CPU RTSP live

Depuis la racine du repo :

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.benchmark --duration 10 --warmup 3
```

Options utiles :

- `--scenarios capture_only overlay_only detector_baseline detector_no_roi detector_low_rate_2fps save_frame_cost plate_pipeline`
- `--output-dir .\data\benchmarks\mon-run`

Sorties :

- un tableau console avec `status`, `wall_fps`, `process_cpu_pct`, `detector_runs`, `vehicle_crops`, `ocr_jobs`, `files_written`
- un resume JSON dans `data/benchmarks/<timestamp>/summary.json`

Definition rapide des scenarios :

- `capture_only` : flux RTSP seul, sans YOLO ni OCR
- `overlay_only` : copie/overlay 4K, sans YOLO ni OCR
- `detector_baseline` : detecteur principal avec la config courante
- `detector_no_roi` : meme detecteur avec ROI desactive
- `detector_low_rate_2fps` : detecteur principal limite a 2 FPS
- `save_frame_cost` : sauvegarde JPEG forcee pour mesurer le cout I/O
- `plate_pipeline` : detecteur principal + second detecteur plaque + OCR

## Logs et stats OCR

Au lancement de `python -m src.main`, deux fichiers courants sont reecrits en place :

- `data/detected_plates.txt`
- `data/fps_stats.txt`

Les historiques detailles sont ecrits dans :

- `data/history/detected_plates-YYYY-MM-DD.txt`
- `data/history/fps_stats-YYYY-MM-DD.txt`

Rotation :

- un nouveau fichier est cree chaque jour selon l'heure locale de la machine.
- pour les plaques, l'historique ecrit au premier verrouillage, au changement de plaque, puis toutes les 30 secondes si la meme plaque reste stable.

Definition du taux OCR :

- `ocr_jobs_processed` = nombre de crops OCR reellement traites.
- `ocr_success_stabilized` = nombre de jobs ayant abouti a une plaque francaise stabilisee.
- `ocr_failure_total` = nombre de jobs n'ayant pas abouti a une plaque stabilisee.
- `ocr_failure_non_french` = texte lu mais hors format FR.
- `ocr_failure_unstable` = candidat FR vu mais pas encore stabilise.
- `ocr_failure_empty` = aucun texte exploitable.

Le taux affiche a l'ecran et dans `detected_plates.txt` :

- `ocr_success_rate_pct = ocr_success_stabilized / ocr_jobs_processed * 100`
- `ocr_failure_rate_pct = ocr_failure_total / ocr_jobs_processed * 100`
