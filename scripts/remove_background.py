import numpy as np
from skimage.color import rgb2lab
from skimage.measure import label
from skimage.morphology import binary_opening, binary_closing, remove_small_objects
from PIL import Image
import cv2
import copy
import os

def create_color_block(color, size=(50, 50)):
    """
    Creates an image block of a given color.

    :param color: The color of the block (RGB format).
    :param size: The size of the block (width, height).
    :return: A PIL Image object filled with the specified color.
    """
    block = Image.new('RGB', size, color=tuple(color.astype(int)))
    return block

def get_representative_background_color(pixels, tolerance=50):  # Tolerance is a fixed value now
    # Extract corner pixels
    corner_pixels = [
        {"pixel": pixels[0, 0, :3], "position": (0,0)},
        {"pixel": pixels[0, -1, :3], "position": (0,-1)},
        {"pixel": pixels[-1, 0, :3], "position": (-1, 0)},
        {"pixel": pixels[-1, -1, :3], "position": (-1,-1)},
    ]

    # Extract pixel values for calculations
    pixel_values = np.array([p["pixel"] for p in corner_pixels])
    mean = np.mean(pixel_values, axis=0)

    # Filter out pixels within the tolerance
    valid_indices = np.all(np.abs(pixel_values - mean) <= tolerance, axis=1)  # Fixed tolerance
    valid_pixels = [corner_pixels[i] for i in range(len(corner_pixels)) if valid_indices[i]]

    # Calculate the average color
    if len(valid_pixels) > 0:
        avg_color = np.mean([p["pixel"] for p in valid_pixels], axis=0)
    else:
        valid_pixels = [{"pixel": pixels[0, 0, :3], "position": (0,0)}]
        avg_color = pixels[0, 0, :3]

    return avg_color, valid_pixels


def remove_background_and_clean_artifacts(image, tolerance=0.01, min_size=64):
    """
    Removes the background of an image using Euclidean distance in LAB color space, labeling,
    and cleans up artifacts such as thin lines and small islands of color.

    :param image_path: Path to the image file.
    :param tolerance: Tolerance for background detection, as a percentage of the max Euclidean distance.
    :param min_size: Minimum size of small objects to be removed.
    :return: A PIL Image object with the background and artifacts removed.
    """
    # Load the image and convert it to RGBA (to add alpha channel)
    pixels = np.array(image)

    # Check if the image has an alpha channel
    if pixels.shape[2] == 3:
        # Add an alpha channel (fully opaque)
        pixels = np.concatenate([pixels, np.full((pixels.shape[0], pixels.shape[1], 1), 255, dtype=np.uint8)], axis=-1)


    # Convert RGB to LAB for better color difference measurement
    lab_pixels = rgb2lab(pixels[:, :, :3] / 255.0)

    # Get the average LAB color of the background
    avg_bg_color_rgb, valid_pixels = get_representative_background_color(pixels)
    avg_bg_color_lab = rgb2lab(np.array([[avg_bg_color_rgb / 255.0]]))[0, 0, :]

    # Calculate the Euclidean distance from the average background color for all pixels in LAB space
    euclidean_distances_lab = np.sqrt(np.sum((lab_pixels - avg_bg_color_lab) ** 2, axis=-1))

    # Calculate the maximum Euclidean distance for the tolerance in LAB color space
    max_euclidean_distance_lab = np.sqrt(100**2 + 255**2 + 255**2)
    tolerance_distance_lab = max_euclidean_distance_lab * tolerance

    # Create a mask where all pixels with a Euclidean distance within the tolerance are marked as background
    background_mask = euclidean_distances_lab <= tolerance_distance_lab

    # Label the regions in the mask
    labeled_mask = label(background_mask, connectivity=1)

    bg_label = labeled_mask[valid_pixels[0]["position"][0], valid_pixels[0]["position"][1]]

    # Create a new mask where only the connected background region is True
    connected_bg_mask = (labeled_mask == bg_label)

    # Clean up artifacts: perform opening and closing operations
    # Opening (erosion followed by dilation) removes small objects
    # Closing (dilation followed by erosion) fills small holes
    clean_mask = binary_opening(connected_bg_mask, np.ones((7,7)))
    clean_mask = binary_closing(clean_mask, np.ones((7,7)))

    # Remove small objects from the mask
    clean_mask = remove_small_objects(clean_mask, min_size=min_size)

    # Set the alpha channel to 0 for background pixels in the cleaned mask
    pixels[clean_mask, 3] = 0

    # Create a new image from the modified pixel array
    new_image = Image.fromarray(pixels)

    # Create a color block for the average background color
    color_block = create_color_block(avg_bg_color_rgb)

    # Return the new image and the color block
    return new_image, color_block

def homomorphic_filter(img):
    # Step 1: Apply the logarithmic transformation
    img_log = np.log1p(np.array(img, dtype="float"))

    # Step 2: Perform the Fourier transform and shift the frequency components
    img_fft = np.fft.fft2(img_log)
    img_fft_shift = np.fft.fftshift(img_fft)

    # Step 3: Create a high-pass filter (correcting the previous low-pass approach)
    rows, cols = img.shape
    radius = 45  # The radius of the high-pass filter
    center = (rows // 2, cols // 2)
    mask = np.ones((rows, cols), dtype="float")
    cv2.circle(mask, center, radius, 0, thickness=-1)
    mask = 1 - mask  # Invert the mask to create a high-pass filter

    # Apply the high-pass filter to the shifted Fourier-transformed image
    img_fft_hp = img_fft_shift * mask

    # Step 4: Inverse shift and apply the inverse Fourier transform
    img_fft_hp_shift = np.fft.ifftshift(img_fft_hp)
    img_ifft = np.fft.ifft2(img_fft_hp_shift)

    # Step 5: Apply the exponential transformation
    img_exp = np.expm1(np.real(img_ifft))

    # Step 6: Normalize the image to bring the pixel values between 0 and 255
    img_out = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(img_out)

def brighten_light_areas(gray_img):
    filtered_img = homomorphic_filter(gray_img)
    
    # Apply Otsu's thresholding
    _, thresholded_img = cv2.threshold(filtered_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded_img
    #blurred = cv2.GaussianBlur(gray_img, (13, 13), 0)
    #adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
 
    #return adaptive_thresh

def isolate_object(img):
    # Check the image mode and convert to RGBA if not already in that format
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Convert PIL image to OpenCV format (BGRA)
    open_cv_image = np.array(img)[:, :, :3]  # Get RGB channels
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    if img.mode == 'RGBA':
        alpha_channel = np.array(img)[:, :, 3]  # Extract the alpha channel
        open_cv_image = cv2.merge((open_cv_image, alpha_channel))  # Add the alpha channel

    gray = cv2.cvtColor(open_cv_image.copy(), cv2.COLOR_BGR2GRAY)
    gray = brighten_light_areas(gray)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    checkpoint = copy.deepcopy(gray)

    # Now apply Canny edge detection
    edges = cv2.Canny(gray, 150, 250)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an initial mask with contour edges
    initial_mask = np.zeros_like(gray)
    cv2.drawContours(initial_mask, contours, -1, 255, thickness=2)

    # Define the kernel size for the morphological operations
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform dilation followed by erosion (closing) on the initial mask
    mask = cv2.dilate(initial_mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    debug_mask = mask.copy()
    # Use flood fill to fill the background
    flood_fill_mask = mask.copy()
    # Invert the flood_fill_mask for the flood fill operation
    inverted_flood_fill_mask = cv2.bitwise_not(flood_fill_mask)

    # Invert the initial mask for the flood fill operation
    inverted_initial_mask = cv2.bitwise_not(initial_mask)

    # Create a mask that is 2 pixels larger than the source image for flood fill
    h, w = inverted_initial_mask.shape[:2]
    flood_fill_mask = np.zeros((h+2, w+2), np.uint8)

    # Apply flood fill with gray color (128)
    cv2.floodFill(inverted_initial_mask, flood_fill_mask, (0,0), 128)

    # Create a mask from the flood-filled image
    # Areas that are not gray (128) are part of the object
    gray_flood_fill_color = 128
    object_mask = cv2.inRange(inverted_initial_mask, 0, gray_flood_fill_color - 1) | cv2.inRange(inverted_initial_mask, gray_flood_fill_color + 1, 255)

    mask_rgba = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGRA)
    mask_rgba[:, :, 3] = object_mask

    # Apply the mask to the original image
    # Ensure that open_cv_image is in BGRA format if it's not already
    if open_cv_image.shape[2] == 3:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2BGRA)

    result_bgra = cv2.bitwise_and(open_cv_image, mask_rgba)

    # Convert the result to RGBA for PIL compatibility (if needed)
    result_pil = Image.fromarray(cv2.cvtColor(result_bgra, cv2.COLOR_BGRA2RGBA))
    mask_pil = Image.fromarray(cv2.cvtColor(checkpoint, cv2.COLOR_BGRA2RGBA))
    return result_pil, mask_pil

def isolate_foreground(img, near_black_threshold=30, kernel_size=5):
    """
    Isolates the foreground from an image based on near-black edge detection.

    Args:
    - img (str): PIL image file.
    - near_black_threshold (int): Threshold for considering a pixel as near black.
    - kernel_size (int): Size of the kernel for morphological operations.

    Returns:
    - result (numpy.ndarray): Image with the foreground isolated.
    """

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Convert PIL image to OpenCV format (BGRA)
    open_cv_image = np.array(img)[:, :, :3]  # Get RGB channels
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    if img.mode == 'RGBA':
        alpha_channel = np.array(img)[:, :, 3]  # Extract the alpha channel
        open_cv_image = cv2.merge((open_cv_image, alpha_channel))  # Add the alpha channel


    # Convert the image to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Create a mask for near black pixels
    near_black_mask = cv2.inRange(gray, 0, near_black_threshold)
    checkpoint = copy.deepcopy(near_black_mask)
    # Edge detection on the near black mask
    edges = cv2.Canny(near_black_mask, 100, 200)

    # Perform morphological closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Invert the closed edges for flood fill
    invert_closed_edges = cv2.bitwise_not(closed_edges)

    # Flood fill from the corner
    h, w = invert_closed_edges.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(invert_closed_edges, mask, (0,0), 255)

    # Invert back to get the foreground mask
    foreground_mask = cv2.bitwise_not(invert_closed_edges)

    # Apply the mask to isolate the foreground
    foreground_mask = cv2.threshold(foreground_mask, 1, 255, cv2.THRESH_BINARY)[1]
    result = cv2.bitwise_and(open_cv_image, open_cv_image, mask=foreground_mask)

    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
    mask_pil = Image.fromarray(cv2.cvtColor(checkpoint, cv2.COLOR_BGRA2RGBA))
    return result_pil, mask_pil

    return result