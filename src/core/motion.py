import cv2


class MotionDetector:
    """Gate ultra-leger base sur la difference entre frames dans la ROI."""

    def __init__(self, resize_width, diff_threshold, min_area_ratio, keepalive_sec):
        self.resize_width = resize_width
        self.diff_threshold = diff_threshold
        self.min_area_ratio = min_area_ratio
        self.keepalive_sec = keepalive_sec
        self.previous_frame = None
        self.last_motion_at = None

    def _prepare_frame(self, frame):
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_height, frame_width = frame.shape[:2]
        if self.resize_width > 0 and frame_width > self.resize_width:
            resized_height = max(1, int(round(frame_height * (self.resize_width / frame_width))))
            frame = cv2.resize(
                frame,
                (self.resize_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )

        return cv2.GaussianBlur(frame, (5, 5), 0)

    def update(self, frame, current_perf):
        prepared_frame = self._prepare_frame(frame)
        motion_ratio = 0.0
        instant_motion = False

        if self.previous_frame is not None and self.previous_frame.shape == prepared_frame.shape:
            delta = cv2.absdiff(self.previous_frame, prepared_frame)
            _, thresholded = cv2.threshold(
                delta,
                self.diff_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            thresholded = cv2.dilate(thresholded, None, iterations=2)
            moving_pixels = cv2.countNonZero(thresholded)
            motion_ratio = moving_pixels / float(thresholded.size)
            instant_motion = motion_ratio >= self.min_area_ratio
            if instant_motion:
                self.last_motion_at = current_perf

        self.previous_frame = prepared_frame
        recent_motion = (
            self.last_motion_at is not None
            and (current_perf - self.last_motion_at) <= self.keepalive_sec
        )

        return {
            "instant_motion": instant_motion,
            "recent_motion": instant_motion or recent_motion,
            "motion_ratio": motion_ratio,
        }
