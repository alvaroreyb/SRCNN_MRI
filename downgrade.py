

from PIL import Image
import os

def resize_images_in_folder(source_folder, dest_folder, downscale_factor):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):
            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path)
            downscale_size = (int(img.width * downscale_factor), int(img.height * downscale_factor))
            img_downscaled = img.resize(downscale_size, Image.ANTIALIAS)

            img_upscaled = img_downscaled.resize(img.size, Image.NEAREST)
            processed_img_path = os.path.join(dest_folder, filename)
            img_upscaled.save(processed_img_path)

source_folder = r""
dest_folder = r""
scale_factor = 0.5
resize_images_in_folder(source_folder, dest_folder, scale_factor)
