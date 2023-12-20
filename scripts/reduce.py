from PIL import Image

def reduce_image_size(img):
    original_avg_color = average_color(img)
    width, height = img.size

    # Start with a square size that is a quarter of the smaller dimension of the image
    square_size = min(width, height) // 4

    while square_size > 1:
        reduced_img = create_reduced_image(img, square_size)
        reduced_avg_color = average_color(reduced_img)

        if not is_color_difference_significant(original_avg_color, reduced_avg_color):
            break  # Acceptable color difference, stop shrinking

        square_size -= 1

    return reduced_img

def create_reduced_image(img, square_size):
    new_width = img.width // square_size
    new_height = img.height // square_size
    new_img = Image.new('RGBA', (new_width, new_height))

    for i in range(new_width):
        for j in range(new_height):
            central_pixel = img.getpixel((i * square_size + square_size // 2, j * square_size + square_size // 2))
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

def is_color_difference_significant(color1, color2, threshold=1):
    # Calculate the Euclidean distance between colors
    distance = sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5
    return distance > threshold