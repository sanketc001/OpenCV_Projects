import cv2
import numpy as np
import mediapipe as mp
mp_selfie_segmentation = mp.solutions.selfie_segmentation
#help(mp_selfie_segmentation.SelfieSegmentation)
# Show segmentation masks.
BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white
def segmentation(image):
    with mp_selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
        # Convert the BGR image to RGB and process it with MediaPipe Selfie Segmentation.
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Generate solid color images for showing the output selfie segmentation mask.
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)
        return output_image
def blur(image):
    # Blur the image background based on the segementation mask.
    with mp_selfie_segmentation.SelfieSegmentation() as selfie_segmentation:
        # Convert the BGR image to RGB and process it with MediaPipe Selfie Segmentation.
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        blurred_image = cv2.GaussianBlur(image,(55,55),0)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        output_image = np.where(condition, image, blurred_image)
        return output_image
