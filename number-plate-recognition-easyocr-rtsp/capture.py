import os
import threading
import time
from queue import Empty, Queue

import cv2


def initialize_video_stream(source):
    is_rtsp = isinstance(source, str) and source.startswith("rtsp://")
    if is_rtsp and "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    if is_rtsp:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source video : {source}")

    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class FrameGrabber(threading.Thread):
    """Capture the most recent frame in a background thread."""

    def __init__(self, source, queue_size=1):
        super().__init__(daemon=True)
        self.source = source
        self.queue = Queue(maxsize=max(1, int(queue_size)))
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.last_error = None
        self.cap = None

    def run(self):
        try:
            self.cap = initialize_video_stream(self.source)
            self.ready.set()
        except Exception as exc:
            self.last_error = exc
            self.ready.set()
            return

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass

            self.queue.put(frame)

        if self.cap is not None:
            self.cap.release()

    def get_latest(self, timeout=0.5):
        if self.last_error is not None:
            raise self.last_error

        try:
            frame = self.queue.get(timeout=timeout)
        except Empty:
            return None

        while not self.queue.empty():
            try:
                frame = self.queue.get_nowait()
            except Empty:
                break
        return frame

    def stop(self):
        self.stop_event.set()
