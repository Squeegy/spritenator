from PIL import Image
import numpy as np
import cv2

def remove_shadow(pil_img, shadow_threshold=100, low_edge_threshold=50,  high_edge_threshold=100):
    # Convert Pillow image to OpenCV format in BGR
    open_cv_image_bgr = np.array(pil_img.convert('RGB'))[:, :, ::-1]

    # Convert image to grayscale
    gray = cv2.cvtColor(open_cv_image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect edges (presumably including black or almost black outlines)
    edges = cv2.Canny(gray, low_edge_threshold, high_edge_threshold)

    # Dilate the edges a bit to ensure they are not removed with the shadow
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Apply thresholding to identify shadow, excluding edges
    _, thresholded = cv2.threshold(gray, shadow_threshold, 255, cv2.THRESH_BINARY)
    shadow_mask = cv2.bitwise_and(thresholded, thresholded, mask=~edges_dilated)

    # Create an alpha channel for transparency
    b, g, r = cv2.split(open_cv_image_bgr)
    original_alpha = np.array(pil_img.split()[-1])  # Get the original alpha channel from PIL image
    new_alpha = cv2.bitwise_and(original_alpha, 255, mask=~shadow_mask)  # Remove shadow from alpha channel

    # Merge channels including the new alpha
    open_cv_image_with_alpha = cv2.merge((b, g, r, new_alpha))

    # Convert back to Pillow image in RGBA format
    return Image.fromarray(cv2.cvtColor(open_cv_image_with_alpha, cv2.COLOR_BGRA2RGBA))
