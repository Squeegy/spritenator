import numpy as np
from skimage.color import rgb2lab
from skimage.measure import label
from skimage.morphology import binary_opening, binary_closing, remove_small_objects
from PIL import Image

def get_representative_background_color(pixels, tolerance=0.05):
    # Extract corner pixels (top-left, top-right, bottom-left, bottom-right)
    corner_pixels = np.array([
        pixels[0, 0, :3],
        pixels[0, -1, :3],
        pixels[-1, 0, :3],
        pixels[-1, -1, :3]
    ])

    # Calculate mean and standard deviation
    mean = np.mean(corner_pixels, axis=0)
    std = np.std(corner_pixels, axis=0)

    # Filter out pixels that are outside the tolerance (mean ± tolerance * std)
    valid_pixels = corner_pixels[np.all(np.abs(corner_pixels - mean) <= tolerance * std, axis=1)]

    # Calculate the average color from the remaining pixels
    if valid_pixels.shape[0] > 0:
        avg_color = np.mean(valid_pixels, axis=0)
    else:
        # Fallback to the original pixel at (0, 0) if no pixels are valid
        avg_color = pixels[0, 0, :3]

    return avg_color

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
    avg_bg_color_rgb = get_representative_background_color(pixels)
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
    bg_label = labeled_mask[0, 0]

    # Create a new mask where only the connected background region is True
    connected_bg_mask = (labeled_mask == bg_label)

    # Clean up artifacts: perform opening and closing operations
    # Opening (erosion followed by dilation) removes small objects
    # Closing (dilation followed by erosion) fills small holes
    clean_mask = binary_opening(connected_bg_mask, np.ones((3,3)))
    clean_mask = binary_closing(clean_mask, np.ones((3,3)))

    # Remove small objects from the mask
    clean_mask = remove_small_objects(clean_mask, min_size=min_size)

    # Set the alpha channel to 0 for background pixels in the cleaned mask
    pixels[clean_mask, 3] = 0

    # Create a new image from the modified pixel array
    new_image = Image.fromarray(pixels)

    return new_image
