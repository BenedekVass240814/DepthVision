import cv2
import mediapipe as mp
from src.webcam_constants import FACIAL_LANDMARK_WINDOW_NAME

# Set up MediaPipe's Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def detect_facial_landmarks(frame):
    """
    Extracts 2D facial landmark points from a video frame using MediaPipe.

    Args:
        frame (numpy.ndarray): Input frame from the video feed.

    Returns:
        list: A list of landmark coordinate sets, one for each detected face.
              Each set contains (x, y) tuples for each landmark.
    """
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)

    landmark_points = []
    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            coords = [
                (int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0]))
                for pt in face.landmark
            ]
            landmark_points.append(coords)

    return landmark_points


def draw_facial_landmarks(frame, landmarks):
    """
    Overlays small green dots on the detected facial landmarks in the frame.

    Args:
        frame (numpy.ndarray): The frame on which landmarks will be drawn.
        landmarks (list): List of facial landmark coordinate sets.

    Returns:
        numpy.ndarray: The frame with facial landmarks visually marked.
    """
    for face in landmarks:
        for x, y in face:
            cv2.circle(frame, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    return frame
