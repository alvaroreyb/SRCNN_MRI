
import numpy as np
import torch
from torchvision.transforms import functional as Func

"""
Parameters:

image: A NumPy array representing the image. 
half: A boolean flag indicating whether to convert the tensor to half precision (torch.float16). May be used to reduce
computational needs
range_norm: A boolean flag indicating whether to normalize the pixel value range from [0, 1] to [-1, 1]. 
May be used to stabilise learning.
Usage: From image to NCHW
"""
def image2tensor(image: np.ndarray, half: bool,range_norm: bool,) -> torch.Tensor:
    tensor = Func.to_tensor(image)
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)
    if half:
        tensor = tensor.half()
    return tensor


"""Parameters:

tensor: A PyTorch tensor representing the image, potentially normalized to the range [-1, 1] and/or in half precision.
half: A boolean indicating whether the tensor is in half precision and needs to be converted back to single precision (torch.float32).
range_norm: A boolean indicating whether the tensor values are normalized to [-1, 1] and need to be scaled back to [0, 1].

Usage: From NCHW to image
"""
def tensor2image(tensor: torch.Tensor, half: bool,range_norm: bool,) -> np.ndarray:
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.float()
    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image 




