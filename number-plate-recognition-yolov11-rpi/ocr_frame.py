from ultralytics import YOLO
import cv2
import easyocr
import re

# Load YOLO model for license plate detection
license_plate_detector = YOLO("./models/license_plate_detector.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'],gpu=False)

# Load video
cap = cv2.VideoCapture("./data/input/traffic.mp4")

# Get frame properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer (only frames with OCR will be written)
output_path = "./data/output/recognized_only.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    show_frame = False  # Flag to decide if we show/write this frame

    # Detect license plates
    results = license_plate_detector.track(frame, persist=True)

    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            try:
                # Convert to grayscale and OCR
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                ocr_result = reader.readtext(resized)

                if ocr_result:
                    raw_text = ocr_result[0][1]
                    # Clean text: keep only A-Z and 0-9
                    plate_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

                    if 4 < len(plate_text) < 10:
                        # Draw box + text
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, plate_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        show_frame = True

            except Exception as e:
                print(f"OCR failed: {e}")

    # Only write/show frame if OCR succeeded
    if show_frame:
        out.write(frame)
        cv2.imshow("Recognized Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

