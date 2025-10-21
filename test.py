import os
import cv2
import numpy as np
import torch
from natsort import natsorted
import configTs
import im2tensor
from metrics import PSNR, SSIM
from model import RBSRCNN  # CNNM1, CNNM2 
from Telegram import TelegramResults 
"""
Initializes the RBSRCNN model and sets it to the specified device with a specific memory format for efficiency.
Loads pre-trained weights into the model from a specified path.
"""
def main() -> None:
    model = RBSRCNN().to(device=configTs.device, memory_format=torch.channels_last); print('Modelo listo')
    weights = torch.load(configTs.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights["state_dict"])
    print(f"Loaded`{os.path.abspath(configTs.model_path)}` weights successfully.")
    results_dir = os.path.join("results", "test", configTs.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model.eval()
    psnr_metrica = PSNR()
    ssim_metrica = SSIM()
    """
    Sets the metrics to the specified device, ensuring non-blocking operations for efficiency.
    """
    psnr = psnr_metrica.to(device=configTs.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim_metrica.to(device=configTs.device, memory_format=torch.channels_last, non_blocking=True)

    psnr_list = []
    ssim_list = []  

    file_names = natsorted(os.listdir(configTs.lr_dir))
    filenum = len(file_names)

    for i in range(filenum):
        lr_image_path = os.path.join(configTs.lr_dir, file_names[i])
        sr_image_path = os.path.join(configTs.sr_dir, file_names[i])
        hr_image_path = os.path.join(configTs.hr_dir, file_names[i])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  

        lr_tensor = im2tensor.image2tensor(lr_image, True, False).unsqueeze_(0)
        hr_tensor = im2tensor.image2tensor(hr_image, True, False).unsqueeze_(0)

        lr_tensor = lr_tensor.to(device=configTs.device, memory_format=torch.channels_last, non_blocking=True)
        hr_tensor = hr_tensor.to(device=configTs.device, memory_format=torch.channels_last, non_blocking=True)
        """
        Generates the SR image by passing the LR tensor through the model,
        """
        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp_(0, 1)

        sr_image = im2tensor.tensor2image(sr_tensor, False, True)
        cv2.imwrite(sr_image_path + ".png", sr_image)
        """
        Converts the SR tensor back to an image format and saves it as a PNG file to the specified path.
        """

        psnr_val = psnr(sr_tensor, hr_tensor).item()
        ssim_val = ssim(sr_tensor, hr_tensor).item()
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    # Calculate standard deviations
    psnr_std = np.std(psnr_list)
    ssim_std = np.std(ssim_list)

    # Calculate averages
    psnr_metrics = np.mean(psnr_list)
    ssim_metrics = np.mean(ssim_list)

    print(f"PSNR: {psnr_metrics:6.4f} (std: {psnr_std:6.4f}) \n"
          f"SSIM: {ssim_metrics:4.4f} (std: {ssim_std:4.4f})")

    TelegramResults("Results (PSNR, SSIM)", psnr_metrics, ssim_metrics, psnr_std, ssim_std)  

if __name__ == "__main__":
    main()