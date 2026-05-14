import cv2


def draw_fps_info(frame, fps, min_fps, max_fps):
    fps_text = f"FPS: {fps:.1f} | Min: {min_fps:.1f} | Max: {max_fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (0, 255, 0)
    bg_color = (0, 0, 0)
    (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
    x, y = frame.shape[1] - text_width - 30, text_height + 20
    cv2.rectangle(
        frame,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + 10),
        bg_color,
        -1,
    )
    cv2.putText(frame, fps_text, (x, y), font, font_scale, text_color, font_thickness)
    return frame
