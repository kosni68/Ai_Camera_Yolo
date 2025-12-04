from ultralytics import YOLO
import cv2
import easyocr

# Load the trained YOLO model for license plate detection
license_plate_detector = YOLO("./models/license_plate_detector.pt")

# Initialize EasyOCR (English + French if needed)
reader = easyocr.Reader(['en', 'fr'])

# Load the input video
cap = cv2.VideoCapture("./data/input/traffic.mp4")

# Get video frame dimensions and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and output video writer
output_path = "./data/output/annotated_traffic.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = license_plate_detector.track(frame, persist=True)

    for result in results:
        for bbox in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])

            # Draw a rectangle around the detected license plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract the region of interest (ROI) for OCR
            roi = frame[y1:y2, x1:x2]

            # === Preprocessing for OCR ===
            try:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Perform OCR on the preprocessed image
                ocr_result = reader.readtext(thresh)

                # Display the recognized text
                if ocr_result:
                    text = ocr_result[0][1]
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"OCR failed: {e}")

    # Write the annotated frame to the output video
    out.write(frame)

    # Display the annotated frame
    cv2.imshow("YOLO + OCR", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

