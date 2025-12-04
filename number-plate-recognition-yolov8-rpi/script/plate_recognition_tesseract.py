import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import pytesseract
import numpy as np
import time
import re

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9,-1], [-1, -1, -1]])  # Simple sharpening kernel
    return cv2.filter2D(image, -1, kernel)

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=2, fy=2)  # Upscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Gaussian blur
    # Alternatively, you could try other denoising techniques
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return thresh

def is_french_plate(text):
    clean = text.replace(" ", "").replace("-", "").upper()
    pattern = r'^([A-Z]{2}\d{3}[A-Z]{2}|\d{4}[A-Z]{2}\d{2}|\d{3}[A-Z]{2}\d{2})$'
    return re.match(pattern, clean) is not None

def format_french_plate(text):
    clean = text.replace(" ", "").replace("-", "").upper()
    # Format AA123AA ? AA-123-AA
    if re.match(r'^[A-Z]{2}\d{3}[A-Z]{2}$', clean):
        return f"{clean[:2]}-{clean[2:5]}-{clean[5:]}"
    # Format 1234AB56 ? 1234 AB 56
    elif re.match(r'^\d{4}[A-Z]{2}\d{2}$', clean):
        return f"{clean[:4]} {clean[4:6]} {clean[6:]}"
    # Format 123AB45 ? 123 AB 45
    elif re.match(r'^\d{3}[A-Z]{2}\d{2}$', clean):
        return f"{clean[:3]} {clean[3:5]} {clean[5:]}"
    else:
        return text  # Return original if no known format

# --- Component Initialization ---

def initialize_camera():
    camera = Picamera2()
    camera.preview_configuration.main.size = (4608, 2592)
    camera.preview_configuration.main.format = "RGB888"
    camera.preview_configuration.align()
    camera.configure("preview")
    camera.start()
    return camera

def load_yolo_model():
    # Display the model selection menu
    print("Choose the YOLO model to use:")
    print("1: YOLO NCNN (license_plate_detector_ncnn_model)")
    print("2: YOLO PyTorch (license_plate_detector.pt)")
    print("3: YOLO PyTorch (yolov8n.pt)")
    print("4: YOLO NCNN (yolov8n.pt)")

    # Get user input for model choice
    choice = input("Enter the number of the selected model: ")

    # Load the model based on user's choice
    if choice == "1":
        print("Loading YOLO NCNN model...")
        model = YOLO("./models/license_plate_detector_ncnn_model")
        return model
    elif choice == "2":
        print("Loading YOLO PyTorch (license_plate_detector.pt)...")
        model = YOLO("./models/license_plate_detector.pt")
        return model
    elif choice == "3":
        print("Loading YOLO PyTorch (yolov8n.pt)...")
        model = YOLO("./models/yolov8n.pt")
        return model
    elif choice == "4":
        print("Loading YOLO NCNN model...")
        model = YOLO("./models/yolov8n_ncnn_model")
        return model
    else:
        print("Invalid choice.")
        sys.exit(1)

# --- OCR with Tesseract ---

def run_tesseract_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = preprocess_for_ocr(image)
    gray = sharpen_image(image)

    # Resize for better OCR accuracy
    resized = cv2.resize(gray, (0, 0), fx=2, fy=2)

    # Tesseract config: OEM 3 (default), PSM 7 (single line)
    #custom_config = r'--oem 3 --psm 7'
    custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    
    print("Running OCR with Tesseract")
    text = pytesseract.image_to_string(resized, config=custom_config)
    return text.strip()
    
def analyze_with_second_model(frame, model):
    print("Running secondary model prediction")
    results = model.predict(frame, imgsz=320)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size > 0:
            return plate_crop
    return np.array([])

def draw_fps_info(frame, fps, min_fps, max_fps):
    """
    Draws FPS information on the given frame with background for better visibility.
    """
    fps_text = f"FPS: {fps:.1f} | Min: {min_fps:.1f} | Max: {max_fps:.1f}"

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5  # Increased for better visibility
    font_thickness = 4
    text_color = (0, 255, 0)
    bg_color = (0, 0, 0)

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
    x, y = frame.shape[1] - text_width - 30, text_height  # top-right corner with padding

    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + 10),
        bg_color,
        -1  # Filled rectangle
    )

    # Draw the FPS text
    cv2.putText(
        frame,
        fps_text,
        (x, y),
        font,
        font_scale,
        text_color,
        font_thickness
    )

    return frame
        
# --- Main Processing Loop ---

def main():
    camera = initialize_camera()
    
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 1280, 720)
    
    
    cv2.imshow("Camera")
    
    return
            
    #model = load_yolo_model()    
    model = YOLO("./models/yolov8n_ncnn_model")
    
    # Load model
    license_plate_detector_model = YOLO("./models/license_plate_detector.pt")

    # Open file to log detected plates
    log_file_path = "./data/detected_plates.txt"
    
    # Open file to log FPS stats
    fps_log_file_path = "./data/fps_stats.txt"
    fps_log_interval = 10  # seconds
    last_fps_log_time = time.time()
    
    fps_history = []   
    last_plate = None
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
    video_output_path = "./data/raw_video.mp4"
    frame_width = 1280
    frame_height = 720
    out = cv2.VideoWriter(video_output_path, fourcc, 20.0, (frame_width, frame_height))


    with open(log_file_path, "a") as log_file, open(fps_log_file_path, "a") as fps_log_file:

        while True:
            print("Capturing image")
            frame = camera.capture_array()
            
            # Rotate the frame by 180 degrees
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Write the raw frame to video
            #out.write(frame)

            print("Running YOLO prediction")
            results = model.predict(frame, imgsz=320)[0]
            annotated_frame = results.plot()

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue
                    
                cv2.imshow("Cropped Plate", plate_crop)
                refined_crop = analyze_with_second_model(plate_crop, license_plate_detector_model)
                if refined_crop.size == 0:
                    continue
                    
                text = run_tesseract_ocr(refined_crop)
                if text:
                    if is_french_plate(text):
                        formatted = format_french_plate(text)
                        if formatted != last_plate:
                            print(f"[VALID] French plate detected: {formatted}")
                            last_plate = formatted

                            # Write to log
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            log_file.write(f"[{timestamp}] {formatted}\n")
                            log_file.flush()
                    else:
                        print(f"[IGNORED] Non-French plate format: {text}")
                        
                # Draw text below the rectangle
                x1=0
                y2=10
                cv2.putText(refined_crop,text,(x1, y2),  # BELOW the bounding box
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0, 255, 0),2)
                            
                cv2.imshow("Plate", refined_crop)

            inference_time = results.speed.get("inference", 1)
            fps = 1000 / inference_time
            fps_history.append(fps)

            # Keep the list short to avoid memory overflow
            if len(fps_history) > 1000:
                fps_history.pop(0)

            min_fps = min(fps_history)
            max_fps = max(fps_history)
            
            # Log FPS every 10 seconds
            current_time = time.time()
            if current_time - last_fps_log_time >= fps_log_interval:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                fps_log_file.write(f"[{timestamp}] Min FPS: {min_fps:.1f} | Max FPS: {max_fps:.1f}\n")
                fps_log_file.flush()

                # Reset FPS history
                fps_history = []
                last_fps_log_time = current_time
                
            annotated_frame = draw_fps_info(annotated_frame, fps, min_fps, max_fps)
            

            cv2.imshow("Camera", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

# --- Entry Point ---
if __name__ == "__main__":
    main()
