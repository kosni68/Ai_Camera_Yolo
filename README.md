
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
libcamera-vid -t 0 --width 1280 --height 720 --codec yuv420 --inline -n -o - | ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 1280x720 -i - -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:8554/mystream
```

```bash
libcamera-vid -t 0 --width 4608 --height 2592 --framerate 14 \
  --codec h264 --inline -n --libav-format h264 -o - | \
ffmpeg -re -f h264 -i - -c:v copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/mystream2
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
