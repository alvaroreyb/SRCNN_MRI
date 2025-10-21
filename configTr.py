
import random
import numpy as np
import torch
from torch.backends import cudnn as cuda
"""
Usage: Easy configuration changes. Simple to be modifying the train.py file

device_select = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda.benchmark = True if torch.cuda.is_available() else False
In case we want automatic PyTorch cuda device selection
"""


device_select = torch.device("cuda",0)
cuda.benchmark = True
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
upscale_factor = 2
nombre = "SRCNN_KNEEMRIARB" #Expertiment name

pfreq= 10000
train_dir = r""
val_lr_dir = r""
val_hr_dir = r""

image_size = 32
batch_size = 16
num_workers = 4
loss = "MSE" #MAE
epochs = 250
model_lr = 1e-4
model_momentum = 0.9
model_weight_decay = 1e-4


