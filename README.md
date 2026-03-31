
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

## Windows setup pour `yolo_with_stream`

Pour preparer l'environnement Windows et lancer la detection de plaques, execute ces commandes depuis la racine du repo :

```powershell
powershell -ExecutionPolicy Bypass -File .\yolo_with_stream\setup_env.ps1
.\yolo_with_stream\.venv\Scripts\Activate.ps1
python .\yolo_with_stream\plate_recognition_tesseract.py
```

Si tu es deja dans le dossier `yolo_with_stream`, utilise plutot :

```powershell
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
python .\plate_recognition_tesseract.py
```

Notes utiles :

- Le script cree ou reutilise `yolo_with_stream/.venv`.
- Python 3.12 est requis.
- Tesseract est installe automatiquement via `winget` s'il n'est pas deja present.
- Pour installer les dependances optionnelles EasyOCR : `python -m pip install -r .\yolo_with_stream\requirements-optional.txt`
- Si `tesseract` n'est pas reconnu juste apres l'installation, ouvre un nouveau terminal PowerShell.
- Les reglages runtime du script sont charges depuis `yolo_with_stream/config.json`.
- `video_display_enabled` dans `yolo_with_stream/config.json` active ou coupe toutes les fenetres OpenCV du script.
- `overlay_enabled` coupe les overlays 4K du flux principal quand tu veux mesurer uniquement la detection sans dessin.
- `ocr_enabled` active ou coupe tout le pipeline OCR.
- `secondary_plate_detector_enabled` active ou coupe le second modele YOLO qui affine le crop plaque avant OCR.
- `fps_limit` dans `yolo_with_stream/config.json` plafonne la boucle principale du script.
- `detector_fps_limit` dans `yolo_with_stream/config.json` cadence les inferences YOLO pour reduire la charge CPU tout en gardant un affichage fluide.
- `roi_enabled`, `roi_x`, `roi_y`, `roi_width` et `roi_height` dans `yolo_with_stream/config.json` permettent de limiter l'analyse du modele principal a une zone rectangulaire normalisee.

## Benchmark CPU RTSP live

Depuis la racine du repo :

```powershell
.\yolo_with_stream\.venv\Scripts\Activate.ps1
python .\yolo_with_stream\benchmark_rtsp.py --duration 10 --warmup 3
```

Options utiles :

- `--scenarios capture_only overlay_only detector_baseline detector_no_roi detector_low_rate_2fps save_frame_cost plate_pipeline`
- `--output-dir .\yolo_with_stream\data\benchmarks\mon-run`

Sorties :

- un tableau console avec `status`, `wall_fps`, `process_cpu_pct`, `detector_runs`, `vehicle_crops`, `ocr_jobs`, `files_written`
- un resume JSON dans `yolo_with_stream/data/benchmarks/<timestamp>/summary.json`

Definition rapide des scenarios :

- `capture_only` : flux RTSP seul, sans YOLO ni OCR
- `overlay_only` : copie/overlay 4K, sans YOLO ni OCR
- `detector_baseline` : detecteur principal avec la config courante
- `detector_no_roi` : meme detecteur avec ROI desactive
- `detector_low_rate_2fps` : detecteur principal limite a 2 FPS
- `save_frame_cost` : sauvegarde JPEG forcee pour mesurer le cout I/O
- `plate_pipeline` : detecteur principal + second detecteur plaque + OCR, avec statut `skipped_no_vehicle_crop` s'il n'y a pas de vehicule exploitable pendant la mesure

## Logs et stats OCR

Au lancement de `python .\yolo_with_stream\plate_recognition_tesseract.py`, deux fichiers "courants" sont maintenant reecrits en place :

- `yolo_with_stream/data/detected_plates.txt`
- `yolo_with_stream/data/fps_stats.txt`

Ces fichiers restent compacts et contiennent uniquement l'etat de session le plus recent.

Les historiques detailles sont ecrits dans :

- `yolo_with_stream/data/history/detected_plates-YYYY-MM-DD.txt`
- `yolo_with_stream/data/history/fps_stats-YYYY-MM-DD.txt`

Rotation :

- un nouveau fichier est cree chaque jour selon l'heure locale de la machine.
- pour les plaques, l'historique detaille ecrit au premier verrouillage, au changement de plaque, puis toutes les 30 secondes si la meme plaque reste stable.

Definition du taux OCR :

- `ocr_jobs_processed` = nombre de crops OCR reellement traites.
- `ocr_success_stabilized` = nombre de jobs ayant abouti a une plaque francaise stabilisee.
- `ocr_failure_total` = nombre de jobs n'ayant pas abouti a une plaque francaise stabilisee.
- `ocr_failure_non_french` = texte lu mais hors format FR.
- `ocr_failure_unstable` = candidat FR vu mais pas encore stabilise.
- `ocr_failure_empty` = aucun texte exploitable.

Le taux affiche a l'ecran et dans `detected_plates.txt` est calcule par session :

- `ocr_success_rate_pct = ocr_success_stabilized / ocr_jobs_processed * 100`
- `ocr_failure_rate_pct = ocr_failure_total / ocr_jobs_processed * 100`
