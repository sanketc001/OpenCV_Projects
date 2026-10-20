# OpenCV & MediaPipe Projects

Real-time computer vision experiments with OpenCV, MediaPipe and dlib: face detection and
recognition, face mesh, hand and pose tracking, segmentation and perspective correction,
plus an ESP32-CAM streaming server for running them on a cheap wireless camera.

## Projects

| Script | What it does |
|--------|--------------|
| `beginner.py`, `Color_Spaces.py` | Webcam capture basics and colour-space conversions (BGR, HSV, grayscale) |
| `detection.py` | Face and eye detection with Haar cascades |
| `recognition.py` | Face recognition against a reference image with `face_recognition` (dlib embeddings) |
| `facerecognition.py` | Attendance-style recognizer: encodes a folder of known faces, matches live frames and logs name + timestamp |
| `meshface.py`, `fullshortface.py` | MediaPipe Face Mesh (468 landmarks) and short/full-range face detection |
| `posegraph3dhands.py` | Hand landmark tracking with MediaPipe Hands |
| `mediapipe_pose.py` | Body pose estimation with joint-angle calculation |
| `mediapipe_holistic.py` | Face, hands and pose together with MediaPipe Holistic |
| `face_hand_pose_combined.py` | Reusable face-mesh detector class combined with hand and pose pipelines in one loop |
| `segmentationblurselfie.py` | Selfie segmentation with background blur |
| `warp perspective card.py` | Perspective transform to flatten a card photographed at an angle |
| `ESP32/` | ESP32-CAM `CameraWebServer` firmware, used as a network video source for the scripts above |
| `HandGestureControl_SixthSense/` | Starting point for a gesture-controlled "Sixth Sense" interface (camera capture loop) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python mediapipe_holistic.py
```

`face_recognition` needs dlib; on Windows install a prebuilt dlib wheel matching your Python
version first. Most scripts read from webcam index 0 or 1; change `cv2.VideoCapture(...)`
to point at another camera or the ESP32-CAM stream URL.

## Notes

These are learning projects written while exploring classical CV and MediaPipe; they are
single-file scripts without tests. The more complete applied work is in my other
repositories.
