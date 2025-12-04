import numpy as np


def analyze_with_second_model(frame, model):
    results = model.predict(frame, imgsz=320)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size > 0:
            return plate_crop
    return np.array([])
