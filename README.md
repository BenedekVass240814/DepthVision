# 🎭 RealSense Face Filters with Gesture Control

A real-time, gesture-controlled face filter application using Intel RealSense cameras, OpenCV, and MediaPipe. This project enables users to apply fun and functional face filters: like mustaches, sunglasses, blur effects, and depth vision—using nothing but facial and hand gestures.

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
git clone https://github.com/BenedekVass240814/DepthVision.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```py
python main.py
```

## Demo

You can view the demo of the project below:

[Watch the demo](demo.mp4)
