from PIL import Image

def enlarge(img, scale_factor):
    """
    Enlarges an image by a given scale factor using the NEAREST filter for a blocky result.
    
    :param img: PIL Image object to be enlarged.
    :param scale_factor: Factor by which the image will be enlarged.
    :return: Enlarged PIL Image object.
    """
    new_size = int(img.width * scale_factor), int(img.height * scale_factor)
    enlarged_img = img.resize(new_size, Image.NEAREST)
    return enlarged_img
