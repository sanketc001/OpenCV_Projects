import cv2
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection

#help(mp_face_detection.FaceDetection)
# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
def short(image):
  with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
      # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
      results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      annotated_image = image.copy()
      if results.detections is not None:
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
      return annotated_image

  # Run MediaPipe Face Detection with full range model.
def full(image):
  with mp_face_detection.FaceDetection(min_detection_confidence=0.5,model_selection=1) as face_detection:
      # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
      results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      annotated_image = image.copy()
      if results.detections is not None:
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
      return annotated_image