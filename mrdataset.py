import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import im2tensor


__all__ = ["TR_VAL_MRIData","TS_MRIData"]

"""
Parameters
image_dir: The directory containing the MRI images.
image_size: The target size to which the images needs to be.
upscale_factor: The factor by which the images will be downscaled and then upscaled to create low-resolution versions.
Usage: This class is intended for loading and preprocessing MRI data for training (Train) purposes.
"""
class TR_VAL_MRIData(Dataset):

    def __init__(self, image_dir: str, image_size: int, upscale_factor: int) -> None:
        super(TR_VAL_MRIData, self).__init__()
        self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in os.listdir(image_dir)]
        self.image_size = image_size
        self.upscale_factor = upscale_factor


    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        try:
            image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        except:
            print(self.image_file_names[batch_index])
            print("Error reading the image")

        lr_image = downrs(image, self.upscale_factor)
        lr_tensor = im2tensor.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = im2tensor.image2tensor(image, range_norm=False, half=False)

        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)
"""
test_lr_image_dir: Directory containing low-resolution MRI test images.
test_hr_image_dir: Directory containing high-resolution MRI test images.
upscale_factor: The factor by which the HR images have been downscaled to create the LR images.
Usage: This class is designed for loading and preprocessing MRI data for testing purposes.
"""

class TS_MRIData(Dataset):

    def __init__(self, test_lr_image_dir: str, test_hr_image_dir: str, upscale_factor: int) -> None:
        super(TS_MRIData, self).__init__()
        self.lr_image_file_names = [os.path.join(test_lr_image_dir, x) for x in os.listdir(test_lr_image_dir)]
        self.hr_image_file_names = [os.path.join(test_hr_image_dir, x) for x in os.listdir(test_hr_image_dir)]
        self.upscale_factor = upscale_factor

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        image_path = self.image_file_names[batch_index]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise IOError(f"Error reading the image at {image_path}")
        
        image = image.astype(np.float32) / 255.0
        lr_image = downrs(image, self.upscale_factor)
        lr_tensor = im2tensor.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = im2tensor.image2tensor(image, range_norm=False, half=False)

        return {"lr": lr_tensor, "hr": hr_tensor}
    

    def __len__(self) -> int:
        return len(self.lr_image_file_names)
"""
Usage: creates downgraded LR MRI using MATLAB type of functions
"""
def downrs(image, upscale_factor: int):
    pil_image = Image.fromarray(np.uint8(image * 255))
    new_width = int(pil_image.width / upscale_factor)
    new_height = int(pil_image.height / upscale_factor)
    downscaled_image = pil_image.resize((new_width, new_height), Image.BICUBIC)
    upscaled_image = downscaled_image.resize(pil_image.size, Image.BICUBIC)
    result_image = np.asarray(upscaled_image).astype(np.float32) / 255.0
    
    return result_image


"""
Function for downgrade using bicubic interpolation.
def downrs(image, upscale_factor:int):
    new_width = int(image.shape[1] / upscale_factor)
    new_height = int(image.shape[0] / upscale_factor)
    downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    upscaled_image = cv2.resize(downscaled_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    return upscaled_image
"""