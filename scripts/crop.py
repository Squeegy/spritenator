from PIL import Image

def crop_transparency(img):
    # Your cropping logic goes here
    # For example, find the bounding box of the non-transparent area and crop to that box
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img
