import cv2
import numpy as np
from src.webcam_constants import (
    BLUR_KERNEL_SIZE,
    SUNGLASSES_IMAGE_PATH,
    MUSTACHE_IMAGE_PATH,
)

def apply_blur_filter(frame, landmarks):
    """
    Applies a blur effect specifically to facial regions based on detected landmarks.

    Args:
        frame (np.ndarray): Current frame from the webcam.
        landmarks (list): List of facial landmarks for each detected face.

    Returns:
        np.ndarray: Frame with facial regions blurred.
    """
    if not landmarks:
        return frame

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for face in landmarks:
        face_region = cv2.convexHull(np.array(face))
        cv2.fillConvexPoly(mask, face_region, 255)

    blurred = cv2.GaussianBlur(frame, BLUR_KERNEL_SIZE, 0)
    frame = np.where(mask[:, :, np.newaxis] == 255, blurred, frame)

    return frame

def apply_sunglasses_filter(frame, landmarks):
    """
    Overlays a sunglasses image onto the eye region of each detected face.

    Args:
        frame (np.ndarray): Webcam frame.
        landmarks (list): Facial landmark coordinates.

    Returns:
        np.ndarray: Frame with sunglasses overlaid on detected faces.
    """
    if not landmarks:
        return frame

    sunglasses = cv2.imread(SUNGLASSES_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        print(f"Error: Could not load sunglasses image at {SUNGLASSES_IMAGE_PATH}")
        return frame

    for face in landmarks:
        left_eye, right_eye = face[33], face[263]
        eye_span = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
        sw = int(eye_span * 2.2)
        ar = sunglasses.shape[0] / sunglasses.shape[1]
        sh = int(sw * ar)

        resized = cv2.resize(sunglasses, (sw, sh), interpolation=cv2.INTER_AREA)

        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        angle = -np.degrees(np.arctan2(dy, dx))

        M = cv2.getRotationMatrix2D((sw // 2, sh // 2), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        center = np.mean([left_eye, right_eye], axis=0).astype(int)
        x, y = int(center[0] - sw / 2), int(center[1] - sh / 2)

        y1, y2 = max(0, y), min(frame.shape[0], y + sh)
        x1, x2 = max(0, x), min(frame.shape[1], x + sw)
        roi = frame[y1:y2, x1:x2]
        overlay = rotated[y1 - y:y2 - y, x1 - x:x2 - x]

        for i in range(overlay.shape[0]):
            for j in range(overlay.shape[1]):
                if overlay[i, j, 3] > 0:
                    roi[i, j] = overlay[i, j, :3]

    return frame

def apply_mustache_filter(frame, landmarks):
    """
    Draws a mustache image under the nose for each detected face.

    Args:
        frame (np.ndarray): Current webcam image.
        landmarks (list): Facial landmarks per face.

    Returns:
        np.ndarray: Frame with mustache graphics overlaid.
    """
    if not landmarks:
        return frame

    mustache = cv2.imread(MUSTACHE_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if mustache is None:
        print(f"Error: Could not load mustache image at {MUSTACHE_IMAGE_PATH}")
        return frame

    for face in landmarks:
        nose = face[1]
        left, right = face[61], face[291]
        mouth_span = np.linalg.norm(np.array(right) - np.array(left))
        mw = int(mouth_span * 1.5)
        ar = mustache.shape[0] / mustache.shape[1]
        mh = int(mw * ar)

        resized = cv2.resize(mustache, (mw, mh), interpolation=cv2.INTER_AREA)

        dx, dy = right[0] - left[0], right[1] - left[1]
        angle = -np.degrees(np.arctan2(dy, dx))
        M = cv2.getRotationMatrix2D((mw // 2, mh // 2), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (mw, mh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        center = np.mean([nose, left, right], axis=0).astype(int)
        x, y = int(center[0] - mw / 2), int(nose[1] - mh * 0.2)

        y1, y2 = max(0, y), min(frame.shape[0], y + mh)
        x1, x2 = max(0, x), min(frame.shape[1], x + mw)
        roi = frame[y1:y2, x1:x2]
        overlay = rotated[y1 - y:y2 - y, x1 - x:x2 - x]

        for i in range(overlay.shape[0]):
            for j in range(overlay.shape[1]):
                if overlay[i, j, 3] > 0:
                    roi[i, j] = overlay[i, j, :3]

    return frame

def apply_depth_vision_filter(depth_frame):
    """
    Converts a RealSense depth frame to a color-mapped visualization.

    Args:
        depth_frame (np.ndarray): Depth data as a 2D array (from RealSense).

    Returns:
        np.ndarray: Colorized depth frame for visualization.
    """
    # Normalize and convert to 8-bit for display
    depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
    depth_8bit = np.uint8(depth_normalized)

    # Apply a colormap to visualize depth
    color_depth = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

    return color_depth
