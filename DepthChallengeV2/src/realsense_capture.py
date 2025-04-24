import pyrealsense2 as rs
import cv2
import numpy as np

from src.webcam_constants import (
    EXIT_KEY,
    WINDOW_NAME,
    FRAME_WAIT_KEY,
    FILTER_NONE_KEY,
    FILTER_LANDMARK_KEY,
    FILTER_BLUR_KEY,
    FILTER_SUNGLASSES_KEY,
    FILTER_MUSTACHE_KEY,
    FILTER_DEPTH_KEY,
    MENU_TEXT,
    MENU_POSITION,
    MENU_FONT,
    MENU_FONT_SCALE,
    MENU_FONT_THICKNESS,
    MENU_COLOR,
)

from src.facial_landmark_detection import detect_facial_landmarks, draw_facial_landmarks
from src.face_filters import (
    apply_blur_filter,
    apply_sunglasses_filter,
    apply_mustache_filter,
    apply_depth_vision_filter,
)
from src.gesture_recognition import detect_gestures
from src.hand_landmark_detection import detect_hand_landmarks  # ✅ NEW IMPORT

# Used in eye_closed_counter logic
EYE_AR_CONSEC_FRAMES = 5

MOUTH_CONSEC_FRAMES = 10

def launch_realsense_filter_app():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    hole_filling_filter = rs.hole_filling_filter()

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error: Unable to start RealSense pipeline: {e}")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)

    current_filter = FILTER_NONE_KEY
    eye_closed_counter = 0
    mouth_open_counter = 0

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_filling_filter.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Detect facial landmarks
        landmarks = detect_facial_landmarks(color_image)

        # Detect hand landmarks
        hand_landmarks = detect_hand_landmarks(color_image)

        gestures = {}
        if landmarks:

            # Detect gestures using both face and hand data
            gestures = detect_gestures(color_image, landmarks, hand_landmarks)

            if gestures.get('mouth_open'):
                mouth_open_counter += 1
                if mouth_open_counter >= MOUTH_CONSEC_FRAMES:
                    current_filter = FILTER_SUNGLASSES_KEY
            else:
                mouth_open_counter = 0


            if gestures.get("one_finger_up"):
                current_filter = FILTER_LANDMARK_KEY
            
            if gestures.get("open_palm"):
                current_filter = FILTER_NONE_KEY

            if gestures.get("hand_above_face"):
                current_filter = FILTER_BLUR_KEY
            
            if gestures.get("finger_under_nose") and gestures.get("one_finger_up"):
                current_filter = FILTER_MUSTACHE_KEY
            
            if gestures.get('eyes_closed'):
                eye_closed_counter += 1
                if eye_closed_counter >= EYE_AR_CONSEC_FRAMES:
                    current_filter = FILTER_DEPTH_KEY
            else:
                eye_closed_counter = 0
        else:
            eye_closed_counter = 0


        # Apply selected filter
        if current_filter == FILTER_DEPTH_KEY:
            frame = apply_depth_vision_filter(depth_image)
        else:
            frame = color_image
            if current_filter == FILTER_LANDMARK_KEY:
                frame = draw_facial_landmarks(frame, landmarks)
            elif current_filter == FILTER_BLUR_KEY:
                frame = apply_blur_filter(frame, landmarks)
            elif current_filter == FILTER_SUNGLASSES_KEY:
                frame = apply_sunglasses_filter(frame, landmarks)
            elif current_filter == FILTER_MUSTACHE_KEY:
                frame = apply_mustache_filter(frame, landmarks)

        # Draw menu
        for i, line in enumerate(MENU_TEXT.split("\n")):
            cv2.putText(
                frame,
                line,
                (MENU_POSITION[0], MENU_POSITION[1] + i * 20),
                MENU_FONT,
                MENU_FONT_SCALE,
                MENU_COLOR,
                MENU_FONT_THICKNESS,
            )

        # Resize to fit window and pad with black
        try:
            x, y, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            h_frame, w_frame = frame.shape[:2]
            scale = min(win_w / w_frame, win_h / h_frame)
            new_w, new_h = int(w_frame * scale), int(h_frame * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            padded_frame = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            top = (win_h - new_h) // 2
            left = (win_w - new_w) // 2
            padded_frame[top:top + new_h, left:left + new_w] = resized_frame
            frame_to_show = padded_frame
        except:
            continue

        cv2.imshow(WINDOW_NAME, frame_to_show)

        key = cv2.waitKey(FRAME_WAIT_KEY) & 0xFF
        if key == ord(EXIT_KEY):
            break
        elif key == ord(FILTER_NONE_KEY):
            current_filter = FILTER_NONE_KEY
        elif key == ord(FILTER_LANDMARK_KEY):
            current_filter = FILTER_LANDMARK_KEY
        elif key == ord(FILTER_BLUR_KEY):
            current_filter = FILTER_BLUR_KEY
        elif key == ord(FILTER_SUNGLASSES_KEY):
            current_filter = FILTER_SUNGLASSES_KEY
        elif key == ord(FILTER_MUSTACHE_KEY):
            current_filter = FILTER_MUSTACHE_KEY
        elif key == ord(FILTER_DEPTH_KEY):
            current_filter = FILTER_DEPTH_KEY

    pipeline.stop()
    cv2.destroyAllWindows()
