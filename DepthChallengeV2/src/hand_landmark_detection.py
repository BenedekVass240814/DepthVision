import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def detect_hand_landmarks(frame):
    """
    Detects hand landmarks in the given frame.

    Args:
        frame (np.ndarray): BGR image.

    Returns:
        list: A list of (x, y) tuples representing hand landmarks, or an empty list.
    """
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    landmarks = []
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        height, width = frame.shape[:2]
        landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in hand.landmark]

    return landmarks
