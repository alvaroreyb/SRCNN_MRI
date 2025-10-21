import os
import shutil
import time
import torch
import sys
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import configTr
from mrdataset import TR_VAL_MRIData, TS_MRIData
from prefetch import DataPrefetchGenerator, DataPrefetchLoader, CPU, CUDAGPU
from metrics import PSNR, SSIM
from model import RBSRCNN
from Telegram import TelegramResults
from enum import Enum

def main() -> None:
    """
    Initializate all parameters
    """
    start_epoch = 1
    GPULimit(0.5)
    best_psnr = 0.0
    best_ssim = 0.0
    train_prefetcher, valid_prefetcher = load_dataset()
    
    model = build_model();
    print('Modelo listo')
    
    loss_crit = loss();
    print('Loss function')
    
    optimizer = optimizador(model);
    print('Optimizer listo')
    
    muestras = os.path.join("muestras", configTr.nombre)
    resultados = os.path.join("resultados", configTr.nombre)
    
    for directory in [muestras, resultados]:
        os.makedirs(directory, exist_ok=True)
    Regs = SummaryWriter(os.path.join("samples", "logs", configTr.nombre))
    GradScaler = amp.GradScaler()
    psnr = PSNR()
    ssim = SSIM()
    """
    sets the metrics to the specified device, ensuring non-blocking operations for efficiency.
    """
    psnr_metrica = psnr.to(device=configTr.device_select, memory_format=torch.channels_last, non_blocking=True)
    ssim_metrica = ssim.to(device=configTr.device_select, memory_format=torch.channels_last, non_blocking=True)
    
    """
    Usage: block for each epoch training via train() function. Both metrics needs to be the best in order to save the model
    as "best"
    Create an pth file for each epoch
    """
    for epoch in range(start_epoch, configTr.epochs):
        train(model, train_prefetcher, loss_crit, optimizer, epoch, GradScaler, Regs)
        psnr, ssim = validation(model, valid_prefetcher, epoch, Regs, psnr_metrica, ssim_metrica)
        best_total = psnr > best_psnr and ssim > best_ssim
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        if epoch%10==0:
            TelegramResults("Epoch",epoch, 'PSNR = ', psnr,'SSIM = ', ssim)
        torch.save({"epoch": epoch,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   os.path.join(muestras, f"epoch_{epoch}.pth.tar"))
        if best_total:
            shutil.copyfile(os.path.join(muestras, f"epoch_{epoch}.pth.tar"),
                            os.path.join(resultados, "best.pth.tar"))
        if (epoch + 1) == configTr.epochs:
            shutil.copyfile(os.path.join(muestras, f"epoch_{epoch}.pth.tar"),
                            os.path.join(resultados, "last.pth.tar"))
            
"""
This function is responsible for preparing the data loading mechanisms for both training and validation datasets. 
It uses TR_VAL_MRIData for training and TS_MRIData for validation and the PyTorch DataLoader to handle data batching
and parallel data loading.
"""

def load_dataset() -> [CPU, CUDAGPU]:
    train_data = TR_VAL_MRIData(configTr.train_dir, configTr.image_size, configTr.upscale_factor, "Train")
    valid_data = TS_MRIData(configTr.val_lr_dir, configTr.val_hr_di, configTr.upscale_factor)
    train_dataloader = DataLoader(train_data, configTr.batch_size, shuffle=False, num_workers=configTr.num_workers,
                                  drop_last=True, persistent_workers=True, pin_memory=True, sampler=None)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                                 persistent_workers=True, pin_memory=True, sampler=None)
    train_prefetcher = CUDAGPU(train_dataloader, configTr.device_select)
    valid_prefetcher = CUDAGPU(valid_dataloader, configTr.device_select)
    
    return train_prefetcher, valid_prefetcher

"""
This function instantiates and prepares the model for training or evaluation.
"""
def build_model() -> nn.Module:
    model = RBSRCNN()
    model = model.to(device=configTr.device_select, memory_format=torch.channels_last)
    
    return model

"""
Function for either MSE or MAE loss functions
"""
def loss() -> nn.MSELoss:
    loss_crit = nn.MSELoss()
    loss_crit = loss_crit.to(device=configTr.device_select, memory_format=torch.channels_last)
    return loss_crit

# def loss() -> nn.L1Loss:
#     loss_crit = nn.L1Loss()
#     loss_crit = loss_crit.to(device=configTr.device_select, memory_format=torch.channels_last)
#     return loss_crit

"""
Define the optimizer model. When comparing with SGD it was used as following:
def optimizador(model) -> optim.SGD:
    optimizer = optim.SGD([{"params": model.features.parameters()},
                           {"params": model.map.parameters()},
                           {"params": model.reconstruction.parameters(), "lr": configTr.model_lr * 0.1}],
                          lr=configTr.model_lr,
                          momentum=configTr.momentum,  # Commonly used momentum value
                          weight_decay=configTr.model_weight_decay,
                          nesterov=True)  

    return optimizer   
    
"""

def optimizador(model) -> optim.Adam:
    optimizer = optim.Adam([{"params": model.features.parameters()},
                            {"params": model.map.parameters()},
                            {"params": model.reconstruction.parameters(), "lr": configTr.model_lr * 0.1}],
                           lr=configTr.model_lr,
                           betas=(0.9, 0.99),
                           eps=configTr.adam_eps,
                           weight_decay=configTr.model_weight_decay)

    return optimizer

"""
Function Parameters:
model: The neural network model to be trained.
train_prefetcher: An instance of CUDAGPU for efficiently loading and transferring batches of training data to the GPU.
loss_crit: The loss criterion to evaluate the difference between the predicted (super-resolved) and ground truth high-resolution images.
optimizer: The optimizer for adjusting the model's weights.
epoch: The current epoch number.
GradScaler: An instance of amp.GradScaler for mixed precision training, allowing for faster computation and reduced memory usage.
writer: A SummaryWriter object for logging training metrics to TensorBoard.
"""

def train(model: nn.Module,
          train_prefetcher: CUDAGPU,
          loss_crit: nn.MSELoss,
          optimizer: optim.Adam,
          epoch: int,
          GradScaler: amp.GradScaler,
          writer: SummaryWriter) -> None:
    batches = len(train_prefetcher)
    batch_time = Avrg("Time", ":6.3f")
    data_time = Avrg("Data", ":6.3f")
    losses = Avrg("Loss", ":6.6f")
    progress = ProgressBar(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")
    model.train() #enables behaviors like dropout and batch normalization specific to training.
    curr_batch_data = train_prefetcher.next()
    batch_index = 0 
    train_prefetcher.reset()
    curr_batch = train_prefetcher.next()
    end = time.time()
    while curr_batch is not None:
        data_time.update(time.time() - end)
        lr = curr_batch_data["lr"].to(device=configTr.device_select, memory_format=torch.channels_last, non_blocking=True)
        hr = curr_batch_data["hr"].to(device=configTr.device_select, memory_format=torch.channels_last, non_blocking=True)
        
        model.zero_grad(set_to_none=True)
        with amp.autocast():
             sr = model(lr)
             loss = loss_crit(sr, hr)
        
        GradScaler.scale(loss).backward()
        GradScaler.step(optimizer)
        GradScaler.update()
        
        losses.update(loss.item(), lr.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_index % configTr.pfreq == 0:  
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches)
            progress.display(batch_index)
        
        curr_batch_data = train_prefetcher.next()
        batch_index += 1 
        
"""
Parameters:
model (nn.Module): model to be evaluated. 
data_prefetcher (CUDAGPU): An instance of CUDAGPU, a custom class designed for efficiently loading 
and prefetching batches of data onto the GPU.
epoch (int): The current epoch number. For logging purposes to track the progress over different epochs.
Regs (SummaryWriter): An instance of SummaryWriter from PyTorch's TensorBoard integration. 
It's used to log metrics like PSNR and SSIM for visualization and analysis in TensorBoard.
psnr_metrica (nn.Module), ssim_metrica (nn.Module): Neural network modules or functions that calculate the Peak Signal-to-Noise Ratio (PSNR)
and Structural Similarity Index Measure (SSIM), respectively. 
"""
def validation(model: nn.Module,
               data_prefetcher: CUDAGPU,
               epoch: int,
               Regs: SummaryWriter,
               psnr_metrica: nn.Module,
               ssim_metrica: nn.Module) -> [float, float]:
    model.eval()
    PSNR_est = Avrg("PSNR", ":4.3f")
    SSIM_est = Avrg("SSIM", ":4.4f")
    batch_time = Avrg("Time", ":6.6f")
    progress = ProgressBar(len(data_prefetcher), [batch_time, PSNR_est, SSIM_est], prefix=f"Epoch: [{epoch + 1}] Validation: ")

    data_prefetcher.reset()
    data_batch = data_prefetcher.next()
    batch_index = 0
    with torch.no_grad():
        while data_batch is not None:
            start_time = time.time()

            lr = data_batch["lr"].to(device=configTr.device_select, memory_format=torch.channels_last, non_blocking=True)
            hr = data_batch["hr"].to(device=configTr.device_select, memory_format=torch.channels_last, non_blocking=True)

            sr = model(lr)
            psnr = psnr_metrica(sr, hr)
            ssim = ssim_metrica(sr, hr)

            PSNR_est.update(psnr.item(), lr.size(0))
            SSIM_est.update(ssim.item(), lr.size(0))
            batch_time.update(time.time() - start_time)

            if batch_index % (len(data_prefetcher) // 5) == 0 or batch_index == len(data_prefetcher) - 1:
                progress.display(batch_index)

            data_batch = data_prefetcher.next()
            batch_index += 1

    progress.display_summary()

    Regs.add_scalar("Valid/PSNR", PSNR_est.avg, epoch + 1)
    Regs.add_scalar("Valid/SSIM", SSIM_est.avg, epoch + 1)
    return PSNR_est.avg, SSIM_est.avg



"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
"""
"""
This enumeration defines types of summaries that can be used to describe the statistical 
information collected during the training or validation process.
"""
class Summary(Enum):
    NONE = 0
    Avrg = 1
    SUM = 2
    COUNT = 3
"""
This class is designed to compute and store statistical metrics such as average, sum, and count. 
Tracking any metric over iterations, such as loss or accuracy, during model training or validation.
"""

class Avrg(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.Avrg):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.Avrg:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Non-valid summary type {self.summary_type}")
        return fmtstr.format(**self.__dict__)
    
"""
Class for displaying progress information and metrics during processing 
"""

class ProgressBar(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
"""
A function to restrict GPU memory usage to a fraction of the total available memory. 
For the shared environments and also to prevent out-of-memory errors by limiting the memory footprint of the process.
"""

def GPULimit(fraction):
    if fraction <= 0.5:
        torch.cuda.set_per_process_memory_fraction(fraction, 'cuda')
        print('Memory restricted to', fraction , "Successfully")
    else:
        sys.exit()

if __name__ == "__main__":
    main()
