from PIL import Image

def reduce_image_size(img, fuzziness=0.9):
    original_avg_color = average_color(img)
    width, height = img.size

    # Start with a rectangle size that is a quarter of the image size
    rect_width, rect_height = width // 4, height // 4

    while rect_width > 1 and rect_height > 1:
        reduced_img = create_reduced_image(img, rect_width, rect_height)
        reduced_avg_color = average_color(reduced_img)

        if not is_color_difference_significant(original_avg_color, reduced_avg_color):
            break  # Acceptable color difference, stop shrinking

        # Find the next largest divisors for width and height with fuzziness
        rect_width = find_largest_divisor(width, rect_width, fuzziness)
        rect_height = find_largest_divisor(height, rect_height, fuzziness)

    return reduced_img

def find_largest_divisor(number, max_divisor, fuzziness):
    # Fuzziness tolerance (e.g., 5% of the number)
    tolerance = number * (1 - fuzziness)

    closest_divisor = 1
    smallest_remainder = number

    for divisor in range(max_divisor - 1, 1, -1):
        remainder = number % divisor

        # Check if the remainder is within the tolerance range
        if remainder <= tolerance:
            if remainder < smallest_remainder:
                closest_divisor = divisor
                smallest_remainder = remainder

    return closest_divisor

def create_reduced_image(img, rect_width, rect_height):
    new_width = img.width // rect_width
    new_height = img.height // rect_height
    new_img = Image.new('RGB', (new_width, new_height))

    for i in range(new_width):
        for j in range(new_height):
            central_pixel = img.getpixel((i * rect_width + rect_width // 2, j * rect_height + rect_height // 2))
            new_img.putpixel((i, j), central_pixel)

    return new_img

def average_color(img):
    r, g, b = 0, 0, 0
    for y in range(img.height):
        for x in range(img.width):
            pixel = img.getpixel((x, y))
            r += pixel[0]
            g += pixel[1]
            b += pixel[2]

    num_pixels = img.width * img.height
    return (r // num_pixels, g // num_pixels, b // num_pixels)

def is_color_difference_significant(color1, color2, threshold=15):
    # Calculate the Euclidean distance between colors
    distance = sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5
    return distance > threshold