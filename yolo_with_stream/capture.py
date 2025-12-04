import cv2
import time
import threading
from queue import Queue, Empty


def initialize_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir le flux RTSP : {rtsp_url}")
    return cap


class FrameGrabber(threading.Thread):
    """Capture le flux RTSP en tache de fond pour ne pas bloquer l'inference."""

    def __init__(self, rtsp_url, queue_size=3):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.queue = Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.last_error = None
        self.cap = None

    def run(self):
        try:
            self.cap = initialize_rtsp_stream(self.rtsp_url)
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

        if self.cap:
            self.cap.release()

    def get_latest(self, timeout=0.5):
        if self.last_error:
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
