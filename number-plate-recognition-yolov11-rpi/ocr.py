from ultralytics import YOLO
import cv2
import easyocr  # OCR

# Load model
license_plate_detector = YOLO("./models/license_plate_detector.pt")

# Initialize OCR reader (French and English, modify as needed)
reader = easyocr.Reader(['en', 'fr'])

# Load video
cap = cv2.VideoCapture("./data/input/traffic.mp4")

# Get frame width, height, and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object to save the video
output_path = "./data/output/annotated_traffic.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Read frames and process
ret = True
while ret:
    ret, frame = cap.read()
    
    if ret:    
        # Detect license plates
        results = license_plate_detector.track(frame, persist=True)

        # Annotate frame with detected license plates
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])

                # Draw rectangle around plate
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Extract ROI (Region Of Interest)
                roi = frame[y1:y2, x1:x2]

                # OCR to read plate
                ocr_result = reader.readtext(roi)

                # Display OCR result
                if ocr_result:
                    text = ocr_result[0][1]  # Get the detected text
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the annotated frame
        cv2.imshow("YOLO Tracking", frame)

        # Break loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
out.release()
cap.release()
cv2.destroyAllWindows()

