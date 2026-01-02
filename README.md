
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
Description=Raspberry Pi Camera RTSP Stream
After=network.target

[Service]
Type=simple
ExecStart=/bin/bash -c "libcamera-vid -t 0 --width 1280 --height 720 --codec yuv420 --rotation 180 --inline -n -o - | ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 1280x720 -i - -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:8554/mystream"
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
