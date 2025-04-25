# 🎭 RealSense Face Filters with Gesture Control

A real-time, gesture-controlled face filter application using Intel RealSense cameras, OpenCV, and MediaPipe. This project enables users to apply fun and functional face filters—like mustaches, sunglasses, blur effects, and depth vision—using nothing but facial and hand gestures.

![Demo Preview](DepthChallengeV2/assets/mustache.png)

## 🚀 Features

- 👀 Facial landmark detection
- ✋ Hand gesture recognition
- 🧠 Smart gesture-filter mapping
- 🕶️ Real-time sunglasses, mustache overlays
- 🌫️ Dynamic face blur filter
- 🔍 Depth-based vision filter
- 🎯 Modular, extensible architecture

## 🧰 Tech Stack

**Language:** Python  
**Libraries:** OpenCV, MediaPipe, pyrealsense2, NumPy  
**Hardware:** Intel RealSense Depth Camera (D435, D415 recommended)

## 📁 Project Structure

assets/
├── sunglasses.png
└── mustache.png

src/
├── face_filters.py                # Functions for blur, overlays, depth
├── facial_landmark_recognition.py# Facial landmark detection (MediaPipe)
├── gesture_recognition.py         # Maps gestures to filters
├── hand_landmark_detection.py     # Hand landmark detection
├── realsense_capture.py           # Core camera and frame processing logic
└── webcam_constant.py             # Constants for UI and filters

main.py                            # Entry point for launching the app
requirements.txt                   # Python dependencies
README.md                          # Project documentation


## 🎮 How It Works

Each gesture triggers a filter:

| Gesture                            | Filter Effect              |
|-----------------------------------|----------------------------|
| Open palm                         | Reset / No Filter          |
| One finger up                     | Facial landmarks overlay   |
| Hand raised above face            | Face blur filter           |
| Mouth open                        | Sunglasses overlay         |
| Finger under nose + raised finger| Mustache overlay           |
| Eyes closed                       | Depth-based grayscale view |

Filters are applied in real-time using live camera data and landmark analysis.

## 🛠️ Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/realsense-face-filters.git
cd realsense-face-filters
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```py
python main.py
```

