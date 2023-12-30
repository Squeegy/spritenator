import numpy as np
from skimage.color import rgb2lab
from skimage.measure import label
from skimage.morphology import binary_opening, binary_closing, remove_small_objects, square
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import copy
import os
import itertools

def visualize_candidates(horizontal_candidates, vertical_candidates, inside_pixels):
    """
    Visualize horizontal, vertical candidates, and their intersection.

    Args:
    - horizontal_candidates (numpy.ndarray): Array of horizontal candidates.
    - vertical_candidates (numpy.ndarray): Array of vertical candidates.
    - inside_pixels (numpy.ndarray): Array of valid inside pixels.

    """
    # Initialize an RGB image
    vis_image = np.zeros((*horizontal_candidates.shape, 3), dtype=np.uint8)

    # Set colors: Red for horizontal, Blue for vertical, Purple for intersection
    vis_image[horizontal_candidates == 1] = [255, 0, 0]  # Red
    vis_image[vertical_candidates == 1] = [0, 0, 255]  # Blue
    vis_image[inside_pixels == 1] = [255, 0, 255]  # Purple

    plt.imshow(vis_image)
    plt.title("Candidate Visualization")
    # Save the plot to a file
    plt.savefig('myplot.png')

def find_inside_candidates(line):
    """
    Find 'inside candidates' in a line by scanning from the edges towards the center.

    Args:
    - line (numpy.ndarray): A single row or column from the mask image.

    Returns:
    - numpy.ndarray: An array marking 'inside candidates.'
    """
    left_scan = np.zeros_like(line)
    right_scan = np.zeros_like(line)

    # Scan from left to right
    for i in range(len(line)):
        if line[i]:
            left_scan[i:] = 1
            break

    # Scan from right to left
    for i in reversed(range(len(line))):
        if line[i]:
            right_scan[:i] = 1
            break

    # Mark 'inside candidates'
    inside_candidates = np.bitwise_and(left_scan, right_scan)
    return inside_candidates

def find_inside_pixels(mask):
    """
    Find valid 'inside pixels' by intersecting horizontal and vertical inside candidates.

    Args:
    - mask (numpy.ndarray): The mask image.

    Returns:
    - numpy.ndarray: An array marking valid 'inside pixels.'
    """
    height, width = mask.shape
    horizontal_candidates = np.zeros_like(mask)
    vertical_candidates = np.zeros_like(mask)

    # Find horizontal inside candidates
    for y in range(height):
        horizontal_candidates[y, :] = find_inside_candidates(mask[y, :])

    # Find vertical inside candidates
    for x in range(width):
        vertical_candidates[:, x] = find_inside_candidates(mask[:, x])

    # Intersection of horizontal and vertical candidates
    inside_pixels = np.bitwise_and(horizontal_candidates, vertical_candidates)
    visualize_candidates(horizontal_candidates, vertical_candidates, inside_pixels)
    
    return inside_pixels

def fill_inside_pixels(mask):
    """
    Fill the inside pixels of the mask.

    Args:
    - mask (numpy.ndarray): The mask image.

    Returns:
    - numpy.ndarray: The mask after filling the inside pixels.
    """
    inside_pixels = find_inside_pixels(mask)
    filled_mask = mask.copy()
    filled_mask[inside_pixels == 1] = 255

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

    return kernel_size, min_line_length + 2 #return kernel and block with buffer

def close_gaps_in_line(line, gap_size_threshold):
    """
    Close small gaps within a line.

    Args:
    - line (numpy.ndarray): A single row or column from the mask.
    - gap_size_threshold (int): Maximum size of gaps to close.

    Returns:
    - numpy.ndarray: The line after closing small gaps.
    """
    closed_line = line.copy()
    gap_start = None

    for i in range(1, len(line)):
        if line[i] and not line[i-1]:  # End of a gap
            if gap_start is not None and (i - gap_start) <= gap_size_threshold:
                # Close the gap
                closed_line[gap_start:i] = 1
                gap_start = None
        elif not line[i] and line[i-1]:  # Start of a gap
            gap_start = i

    return closed_line

def close_gaps(mask, gap_size_threshold):
    """
    Close small gaps in a mask.

    Args:
    - mask (numpy.ndarray): The binary mask.
    - gap_size_threshold (int): Maximum size of gaps to close.

    Returns:
    - numpy.ndarray: The mask after closing gaps.
    """
    height, width = mask.shape
    closed_mask = mask.copy()

    # Close gaps horizontally
    for y in range(height):
        closed_mask[y, :] = close_gaps_in_line(mask[y, :], gap_size_threshold)

    # Close gaps vertically
    for x in range(width):
        closed_mask[:, x] = close_gaps_in_line(closed_mask[:, x], gap_size_threshold)

    return closed_mask

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
    kernel_size, block_size = estimate_kernel_size(near_black_mask)
    kernel = square(kernel_size)
    closed_mask = binary_closing(closed_mask, kernel)
    closed_mask = fill_inside_pixels(closed_mask)
    # Close gaps in the checkpoint mask
    
    closed_mask = close_gaps(closed_mask, kernel_size)

    # Ensure the closed_mask is in the correct format
    if closed_mask.dtype != np.uint8:
        closed_mask = closed_mask.astype(np.uint8)


    # Apply the mask to isolate the foreground
    result = cv2.bitwise_and(open_cv_image, open_cv_image, mask=closed_mask)

    # Convert the mask to an alpha channel
    alpha_channel = 255 - closed_mask  # Inverts the mask to create transparency

    # Add the alpha channel to the result
    result_with_alpha = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # Convert to RGB
    result_with_alpha = np.dstack((result_with_alpha, alpha_channel))  # Add alpha channel

    # Convert to PIL Image
    result_pil = Image.fromarray(result_with_alpha)

    return result_pil