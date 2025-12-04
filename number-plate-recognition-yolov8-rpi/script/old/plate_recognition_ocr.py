import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import easyocr 

# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv8 model (NCNN export)
model = YOLO("./models/license_plate_detector_ncnn_model")

# Initialize OCR reader (English)
reader = easyocr.Reader(['en'])

while True:
    print("Capture d'une image")
    frame = picam2.capture_array()

    print("Lancement de la detection YOLO")
    results = model.predict(frame, imgsz=320)

    result = results[0]

    print("Annotation de l'image")
    annotated_frame = result.plot()

    print("Traitement des detections")
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        plate_crop = frame[y1:y2, x1:x2]

        plate_crop = cv2.resize(plate_crop, (0, 0), fx=2, fy=2)

        gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)

        print("Lecture OCR en cours")
        ocr_result = reader.readtext(gray_crop)

        for detection in ocr_result:
            text = detection[1]
            conf = detection[2]
            if conf > 0.3:
                print(f"Plaque detect : {text} (confiance : {conf:.2f})")
                cv2.putText(
                    annotated_frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    inference_time = result.speed['inference']
    fps = 1000 / inference_time
    fps_text = f'FPS: {fps:.1f}'
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(annotated_frame, fps_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    print("Affichage du resultat")
    cv2.imshow("Camera", annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
