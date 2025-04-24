import numpy as np

# Constants for EAR calculation
EAR_THRESHOLD = 0.2
MOUTH_OPEN_THRESHOLD = 10  # In pixels
HAND_ABOVE_FACE_THRESHOLD = 0.2  # Threshold for hand being above the face (relative to nose)

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (A + B) / (2.0 * C)

def is_mouth_open(landmarks):
    top_lip = np.array(landmarks[0][13])   # Inner upper lip
    bottom_lip = np.array(landmarks[0][14])  # Inner lower lip
    left_lip = np.array(landmarks[0][78])   # Left mouth corner
    right_lip = np.array(landmarks[0][308]) # Right mouth corner

    vertical = np.linalg.norm(top_lip - bottom_lip)
    horizontal = np.linalg.norm(left_lip - right_lip)

    ratio = vertical / horizontal if horizontal != 0 else 0
    return ratio > 0.3  # Adjust threshold based on testing

def is_finger_extended(hand, tip_id, pip_id):
    # Ensure that hand is large enough and the landmark ids are valid
    if len(hand) > max(tip_id, pip_id):  # Make sure the indices are within range
        return hand[tip_id][1] < hand[pip_id][1]  # Y of tip above pip (in image coords)
    return False

def detect_hand_gestures(hand_landmarks):
    if not hand_landmarks or len(hand_landmarks) < 21:  # Check if we have at least 21 landmarks
        return False, False  # No valid hand landmarks found

    hand = hand_landmarks  # Use the full list of landmarks for the hand

    # Check finger extensions using tip and pip landmarks
    fingers = [
        is_finger_extended(hand, 8, 6),   # Index
        is_finger_extended(hand, 12, 10), # Middle
        is_finger_extended(hand, 16, 14), # Ring
        is_finger_extended(hand, 20, 18)  # Pinky
    ]

    thumb = hand[4][0] > hand[3][0]  # Simple thumb open check based on X coordinates

    # Check if all fingers are extended for an open palm
    open_palm = thumb and all(fingers)
    # Check if only the index finger is extended for "one finger up"
    one_finger_up = fingers[0] and not any(fingers[1:])

    return open_palm, one_finger_up

def are_eyes_closed(landmarks):
    left_eye = [landmarks[0][i] for i in [33, 160, 158, 133, 153, 144]]
    right_eye = [landmarks[0][i] for i in [362, 385, 387, 263, 373, 380]]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD

def is_hand_above_face(hand, facial_landmarks):
    if not facial_landmarks or len(facial_landmarks[0]) < 152:  # Make sure we have the required facial landmarks
        return False

    # Compare wrist (landmark 0 of hand) with nose (landmark 1 of face)
    wrist_y = hand[0][1]  # Y-coordinate of the wrist
    nose_y = facial_landmarks[0][1][1]  # Y-coordinate of the nose (facial landmark 1)

    # The hand is considered above the face if the wrist's Y-coordinate is above the nose's Y-coordinate
    return wrist_y < nose_y - HAND_ABOVE_FACE_THRESHOLD * nose_y  # Add threshold for flexibility

def is_finger_under_nose(hand, facial_landmarks):
    if not facial_landmarks or len(facial_landmarks[0]) < 1:  # Make sure we have the required facial landmarks
        return False

    # Get Y and X coordinates of the index finger tip (landmark 8)
    finger_tip_y = hand[8][1]  # Y-coordinate of the index finger tip (landmark 8)
    finger_tip_x = hand[8][0]  # X-coordinate of the index finger tip (landmark 8)

    # Get Y and X coordinates of the nose (landmark 1)
    nose_y = facial_landmarks[0][1][1]  # Y-coordinate of the nose (facial landmark 1)
    nose_x = facial_landmarks[0][1][0]  # X-coordinate of the nose (facial landmark 1)

    # Tolerance for horizontal alignment (you can adjust this value if needed)
    HORIZONTAL_TOLERANCE = 50  # This value defines how far the finger can be horizontally from the nose

    # Check if the finger is vertically below the nose and horizontally aligned within tolerance
    is_vertical_under = finger_tip_y > nose_y
    is_horizontal_aligned = abs(finger_tip_x - nose_x) < HORIZONTAL_TOLERANCE

    return is_vertical_under and is_horizontal_aligned  # Both conditions must be true


def detect_gestures(frame, facial_landmarks, hand_landmarks):
    gestures = {
        "mouth_open": False,
        "eyes_closed": False,
        "open_palm": False,
        "one_finger_up": False,
        "hand_above_face": False,
        "finger_under_nose": False
    }

    if facial_landmarks:
        gestures["mouth_open"] = is_mouth_open(facial_landmarks)
        gestures["eyes_closed"] = are_eyes_closed(facial_landmarks)

    open_palm, one_finger_up = detect_hand_gestures(hand_landmarks)
    gestures["open_palm"] = open_palm
    gestures["one_finger_up"] = one_finger_up

    if hand_landmarks:
        gestures["hand_above_face"] = is_hand_above_face(hand_landmarks, facial_landmarks)
        gestures["finger_under_nose"] = is_finger_under_nose(hand_landmarks, facial_landmarks)

    return gestures
