
import torch
import numpy as np
import math
from torch import nn
import torch.nn.functional as F

__all__ = ["PSNR","SSIM"]

def compT(img1: torch.Tensor, img2: torch.Tensor):
    assert img1.shape == img2.shape, "Shapes does not fit"

"""
Usage: This function calculates the PSNR between two images represented as PyTorch tensors.
"""

def Tpsnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    compT(img1,img2)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse.item()))
    return psnr
"""
A PyTorch module wrapper for the Tpsnr function, making it compatible with PyTorch model and loss architecture. 
It could be used as a loss function, but will be used as metric during model training.
"""
class PSNR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
      
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        psnr_metrics = Tpsnr(img1, img2)
        return psnr_metrics

def gaussian_window(size, sigma):
    gauss = torch.Tensor([math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
"""
A PyTorch module for calculating the SSIM index between two images.
"""

class SSIM(nn.Module):
    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1  
        self.window = create_window(self.window_size, self.channel)


    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).type(img1.dtype).to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

"""
Adjusts the Gaussian window to match the channel size and datatype of the input images.
Uses the Gaussian window to convolve with the input images and calculate local means (mu1, mu2), variances (sigma1_sq, sigma2_sq), and covariance (sigma12).
Calculates the SSIMmap using the constants C1 = 0.01^2 and C2= 0.03^2 to stabilize division with small denominators.
Returns the mean of the SSIM map as the overall SSIM value for the input images.
"""

    
    

