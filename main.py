import os
import shutil
from PIL import Image
import traceback
import scripts.remove_background

print("MAIN LOADED")

# Function to process images (replace this with your image processing logic)
def process_image(image_path):
    try:
        # Open the image
        print(f"opening image: {image_path}")
        img = Image.open(image_path)

        # Image processing code
        img, block = scripts.remove_background.remove_background_and_clean_artifacts(img, 0.05)

        # Create the output path within the "sprites" folder
        root, _ = os.path.splitext(image_path)  # Split the path and ignore the original extension
        output_path = f"{root}.png"  # Append '.png' extension
        output_path = os.path.join("sprites", os.path.basename(image_path))

        # Save the processed image to the output folder
        img.save(output_path)
        block.save(os.path.join("sprites", "BACKGROUND" + os.path.basename(image_path)))

        print(f"Processed image saved: {output_path}")

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        filename, line_no, func, text = tb[0]
        # Log errors to a file called "error.out" in the "sprites" folder
        error_message = f"Error processing image '{image_path}': {str(e)}\n"
        error_message += f"Error occurred in file: {filename}, line: {line_no}, function: {func}\n"

        print(error_message)
        with open(os.path.join("sprites", "error.out"), "a") as error_file:
            error_file.write(error_message + "\n")

def main():
    # Create a "sprites" folder to store processed images
    os.makedirs("sprites", exist_ok=True)

    # Get a list of image files in the current directory
    image_files = [f for f in os.listdir() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print("No image files found in the current directory.")
        return

    # Process each image using the imported scripts
    for image_file in image_files:
        print("PROCESSING", image_file)
        input_path = image_file

        # Process the image in memory and save it to the "sprites" folder
        process_image(input_path)

print("starting")
main()