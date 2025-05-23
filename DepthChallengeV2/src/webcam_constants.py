import cv2

# Constants for webcam capture
WEBCAM_INDEX = 0  # Default webcam index
EXIT_KEY = "q"  # Key to press to exit the webcam feed
WINDOW_NAME = "Webcam Feed"  # Name of the window displaying the webcam feed
FRAME_WAIT_KEY = 1  # Delay in milliseconds for each frame

# Constants for facial landmark detection
FACIAL_LANDMARK_WINDOW_NAME = "Facial Landmark Detection"

# Constants for face filters
BLUR_KERNEL_SIZE = (31, 31)  # Kernel size for the blur filter

# Constants for filter selection keys
FILTER_NONE_KEY = "0"
FILTER_LANDMARK_KEY = "1"
FILTER_BLUR_KEY = "2"
FILTER_SUNGLASSES_KEY = "3"
FILTER_MUSTACHE_KEY = "4"
FILTER_DEPTH_KEY = "5"  # Key to trigger the depth vision filter

# Path to assets
SUNGLASSES_IMAGE_PATH = "assets/sunglasses.png"
MUSTACHE_IMAGE_PATH = "assets/mustache.png"

# Constants for on-screen menu
MENU_TEXT = (
    "Open palm for no filter\n"
    "One finger up for facial landmark detection\n"
    "Raise hand for blur filter\n"
    "Open mouth for sunglasses filter\n"
    "Put finger under nose for mustache filter\n"
    "Close eyes for depth vision filter\n"  # Added depth vision option
    "Press 'q' to exit"
)
MENU_POSITION = (10, 30)  # Coordinates for the menu text
MENU_FONT = cv2.FONT_HERSHEY_SIMPLEX
MENU_FONT_SCALE = 0.4
MENU_FONT_THICKNESS = 1
MENU_COLOR = (255, 255, 255)  # White color
