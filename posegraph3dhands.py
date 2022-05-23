import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
#help(mp_hands.Hands)
# Run MediaPipe Hands.
def pose(image):
    with mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.7) as hands:
        # Convert the BGR image to RGB, flip the image around y-axis for correct
        # handedness output and process it with MediaPipe Hands.
        results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
        print(results.multi_handedness)
        image_hight, image_width, _ = image.shape
        annotated_image = cv2.flip(image.copy(), 1)
        for hand_landmarks in results.multi_hand_landmarks:
          # Print index finger tip coordinates.
          print(
              f'Index finger tip coordinate: (',
              f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
              f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
          )
          mp_drawing.draw_landmarks(
              annotated_image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
        return annotated_image
def graph3d(image):
    # Run MediaPipe Hands and plot 3d hands world landmarks.
    with mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.7) as hands:
        # Convert the BGR image to RGB and process it with MediaPipe Hands.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Draw hand world landmarks.
        for hand_world_landmarks in results.multi_hand_world_landmarks:
          mp_drawing.plot_landmarks(
            hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)