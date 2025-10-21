import os
import cv2
import numpy as np
import torch
from natsort import natsorted
import configTs
import im2tensor
from model import RBSRCNN #CNNM1 #CNMM2
"""
Usage: given a LR folder, supperresolute them with the model and model weights in order to obtain
SR results.
"""
def main() -> None:
    model = RBSRCNN().to(device=configTs.device, memory_format=torch.channels_last)
    print('Model loaded successfully.')
    weights = torch.load(configTs.model_path, map_location=configTs.device)
    model.load_state_dict(weights["state_dict"])
    model.eval()

    sr_results_dir = os.path.join("results", "sr_images", configTs.exp_name)
    os.makedirs(sr_results_dir, exist_ok=True)

    lr_image_paths = natsorted(os.listdir(configTs.lr_dir))
    for lr_image_name in lr_image_paths:
        lr_image_path = os.path.join(configTs.lr_dir, lr_image_name)
        print(f"Processing: {lr_image_path}")

        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        lr_tensor = im2tensor.image2tensor(lr_image, False, True).unsqueeze_(0)
        lr_tensor = lr_tensor.to(device=configTs.device, memory_format=torch.channels_last)

        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp_(0, 1)

        sr_image = im2tensor.tensor2image(sr_tensor, False, True)
        # Generalizing file extension replacement for different image formats
        sr_image_name = os.path.splitext(lr_image_name)[0] + '_sr.png'
        sr_image_path = os.path.join(sr_results_dir, sr_image_name)
        cv2.imwrite(sr_image_path, sr_image)
        print(f"Saved SR image to: {sr_image_path}")

if __name__ == "__main__":
    main()

