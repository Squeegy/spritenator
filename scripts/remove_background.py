import numpy as np
from skimage.color import rgb2lab
from skimage.measure import label
from skimage.morphology import binary_opening, binary_closing, remove_small_objects, square
from PIL import Image
import cv2
import copy
import os
import itertools

def fill_between_white_pixels(line):
    """
    Fill in all pixels between two white pixels in a given line.

    Args:
    - line (numpy.ndarray): A single row or column from the mask image.

    Returns:
    - numpy.ndarray: The modified line with pixels filled between white pixels.
    """
    filled_line = line.copy()
    start = end = None
    for i in range(len(line)):
        if line[i] == 255:  # White pixel found
            if start is None:
                start = i
            else:
                end = i
                # Fill in between
                filled_line[start:end+1] = 255
                start = i  # Reset start for next segment
    return filled_line

def fill_mask(mask):
    """
    Apply the fill_between_white_pixels operation across the entire mask.

    Args:
    - mask (numpy.ndarray): The mask image.

    Returns:
    - numpy.ndarray: The mask after filling between white pixels.
    """
    filled_mask = mask.copy()

    # Apply horizontally
    for y in range(mask.shape[0]):
        filled_mask[y, :] = fill_between_white_pixels(mask[y, :])

    # Apply vertically
    for x in range(mask.shape[1]):
        filled_mask[:, x] = fill_between_white_pixels(mask[:, x])

    return filled_mask

def estimate_kernel_size(near_black_mask, noise_threshold=10, scale_factor=1.5):
    """
    Estimates the kernel size for morphological operations based on scanning for the smallest
    continuous line of the black outline that's above a noise threshold.

    Args:
    - near_black_mask (numpy.ndarray): The mask image.
    - noise_threshold (int): The threshold to ignore noise lines.
    
    Returns:
    - kernel_size (int): The estimated kernel size.
    """
    # Initialize the minimum line length to a large value
    min_line_length = np.inf

    # Scan vertically
    for col in range(near_black_mask.shape[1]):
        line = near_black_mask[:, col]
        line_lengths = [len(list(group)) for key, group in itertools.groupby(line) if key == 0]
        valid_lengths = [length for length in line_lengths if length > noise_threshold]
        if valid_lengths:
            min_line_length = min(min_line_length, min(valid_lengths))

    # Scan horizontally
    for row in range(near_black_mask.shape[0]):
        line = near_black_mask[row, :]
        line_lengths = [len(list(group)) for key, group in itertools.groupby(line) if key == 0]
        valid_lengths = [length for length in line_lengths if length > noise_threshold]
        if valid_lengths:
            min_line_length = min(min_line_length, min(valid_lengths))

    # Check if a valid line was found
    if min_line_length == np.inf:
        # No valid line found
        return max(3, noise_threshold + 1)  # Default to noise threshold + 1, or 3, whichever is larger

    # The kernel size is slightly larger than the smallest line found
    kernel_size = int((min_line_length + 2) * scale_factor)

    # Make sure the kernel size is odd to have a central pixel
    if kernel_size % 2 == 0:
        kernel_size += 1

    return kernel_size

def close_gaps_in_line(line, max_gap_size):
    new_line = line.copy()
    for i in range(1, len(line) - max_gap_size):
        if line[i] == 0 and all(line[i + j] == 255 for j in range(1, max_gap_size + 1)):
            new_line[i:i + max_gap_size] = 255
    return new_line

def close_gaps(mask, max_gap_size):
    height, width = mask.shape

    # Close gaps horizontally
    for y in range(height):
        mask[y, :] = close_gaps_in_line(mask[y, :], max_gap_size)

    # Close gaps vertically
    for x in range(width):
        mask[:, x] = close_gaps_in_line(mask[:, x], max_gap_size)

    return mask

def isolate_foreground(img, near_black_threshold=30):
    """
    Isolates the foreground from an image based on near-black edge detection and iterative morphological closing.

    Args:
    - img (PIL.Image): PIL image file.
    - near_black_threshold (int): Threshold for considering a pixel as near black.
    - closing_iterations (int): Number of iterations for morphological closing, with decreasing kernel size.

    Returns:
    - result_pil (PIL.Image): Image with the foreground isolated.
    - mask_pil (PIL.Image): Binary mask of the isolated foreground.
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Convert PIL image to OpenCV format
    open_cv_image = np.array(img)[:, :, :3]  # Get RGB channels
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Create a mask for near black pixels
    near_black_mask = cv2.inRange(gray, 0, near_black_threshold)

    # Perform iterative morphological closing
    closed_mask = near_black_mask
    kernel_size = estimate_kernel_size(near_black_mask)
    kernel = square(kernel_size)
    closed_mask = binary_closing(closed_mask, kernel)
    checkpoint = fill_mask(closed_mask)
    #checkpoint = copy.deepcopy(closed_mask)

    closed_mask = close_gaps(closed_mask, kernel_size)

    # Ensure the closed_mask is in the correct format
    if closed_mask.dtype != np.uint8:
        closed_mask = closed_mask.astype(np.uint8)

    # Invert the closed mask for flood fill
    invert_closed_mask = cv2.bitwise_not(closed_mask)

    # Use flood fill from the corners of the image
    h, w = invert_closed_mask.shape[:2]
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    for corner in [(0,0), (0, w-1), (h-1, 0), (h-1, w-1)]:
        cv2.floodFill(invert_closed_mask, flood_fill_mask, corner, 255)

    # Invert back to get the filled foreground mask
    filled_foreground_mask = cv2.bitwise_not(invert_closed_mask)

    # Apply the mask to isolate the foreground
    result = cv2.bitwise_and(open_cv_image, open_cv_image, mask=filled_foreground_mask)

    # Convert the result and mask to PIL images
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
    mask_pil = Image.fromarray(checkpoint)

    return result_pil, mask_pil