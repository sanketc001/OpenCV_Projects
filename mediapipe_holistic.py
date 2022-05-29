import cv2
import math
import numpy as np
import mediapipe as mp
DESIRED_HEIGHT = 600
DESIRED_WIDTH = 800
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("",img)

def holistic(image):
    name="Image"
    mp_holistic = mp.solutions.holistic
    #help(mp_holistic.Holistic)

    # Import drawing_utils and drawing_styles.
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Run MediaPipe Holistic and draw pose landmarks.
    with mp_holistic.Holistic(
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as holistic:
        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print nose coordinates.
        image_hight, image_width, _ = image.shape
        if results.pose_landmarks:
          print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
          )

        # Draw pose landmarks.
        print(f'Pose landmarks of {name}:')
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_pose_landmarks_style())
        #resize_and_show(annotated_image)
    return annotated_image

    # Run MediaPipe Holistic and plot 3d pose world landmarks.
    with  mp_holistic.Holistic(static_image_mode=True) as holistic:

        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print the real-world 3D coordinates of nose in meters with the origin at
        # the center between hips.
        print('Nose world landmark:'),
        print(results.pose_world_landmarks.landmark[mp_holistic.PoseLandmark.NOSE])

        # Plot pose world landmarks.
        print(f'Pose world landmarks of {name}:')
        #mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Run MediaPipe Holistic with `enable_segmentation=True` to get pose segmentation.
    with mp_holistic.Holistic(
        static_image_mode=True, enable_segmentation=True) as holistic:
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Draw pose segmentation.
        print(f'Pose segmentation of {name}:')
        annotated_image = image.copy()
        red_img = np.zeros_like(annotated_image, dtype=np.uint8)
        red_img[:, :] = (255,255,255)
        segm_2class = 0.2 + 0.8 * results.segmentation_mask
        segm_2class = np.repeat(segm_2class[..., np.newaxis], 3, axis=2)
        annotated_image = annotated_image * segm_2class + red_img * (1 - segm_2class)
    return annotated_image

