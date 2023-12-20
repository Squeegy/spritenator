from PIL import Image
import numpy as np
import cv2

def remove_shadow(pil_img, shadow_threshold = 50):
    # Load the image with Pillow

    # Convert the Pillow image to an OpenCV image
    open_cv_image = np.array(pil_img)
    open_cv_image_bgr = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

    # Convert image to grayscale for shadow detection
    gray = cv2.cvtColor(open_cv_image_bgr, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to identify shadow
    _, thresholded = cv2.threshold(gray, shadow_threshold, 255, cv2.THRESH_BINARY)

    # Use the thresholded image to create an alpha channel
    # Where the shadow is, set alpha to 0 (transparent)
    b, g, r, a = cv2.split(open_cv_image)
    a[thresholded == 0] = 0  # Set alpha to 0 where the shadow is detected

    # Merge back the channels including the new alpha
    open_cv_image = cv2.merge((b, g, r, a))

    # Convert back to Pillow image
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(open_cv_image)
