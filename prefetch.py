import os
import queue
from threading import Thread
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ["CUDAGPU","CPU","DataPrefetchLoader","DataPrefetchGenerator"]
"""
Usage: class designed for efficiently loading data to CPU memory.
"""

class CPU:
    def __init__(self, dataloader, device: torch.device) -> None:
        self.dataloader = dataloader
        self.device = device
        self.iterator = iter(dataloader)

    def next(self):
        try:
            data = next(self.iterator)
            return {k: v.to(self.device, non_blocking=True) for k, v in data.items()}
        except StopIteration:
            return None

    def reset(self):
        self.iterator = iter(self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)
    
"""
Usage: this class specifically handles prefetching data batches to GPU memory. 
"""
class CUDAGPU:

    """
    inite: Sets up an iterator over the provided DataLoader, a CUDA stream for asynchronous data transfers, 
    and initiates the preload of the first data batch.
    
    """
    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()
    """
    Preloads the next batch of data using the CUDA stream for non-blocking data transfer to the GPU.
    """
    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return
        except Exception as e:
            print(f"Error loading data: {e}")
            self.batch_data = None
            return

        with torch.cuda.stream(self.stream):
            self.batch_data = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in self.batch_data.items()}
    """
    Waits for the current CUDA stream operations to complete, returns the preloaded batch, and initiates loading of the next batch.
    """
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.batch_data is None:
            return None
        batch_data = self.batch_data
        self.preload()
        return batch_data
    """
    Resets the data iterator to start from the beginning of the DataLoader.
    """
    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)

class DataPrefetchGenerator:
    def __init__(self, dataloader, device, queue_size=2):
        self.dataloader = dataloader
        self.device = device
        self.queue = queue.Queue(queue_size)
        self.thread = Thread(target=self.load_data)
        self.thread.daemon = True
        self.iterator = iter(dataloader)
        self.started = False

    def start(self):
        if not self.started:
            self.thread.start()
            self.started = True

    def load_data(self):
        while True:
            try:
                data = next(self.iterator)
                data = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in data.items()}
                self.queue.put(data)
            except StopIteration:
                self.queue.put(None)
                break
            except Exception as e:
                print(f"Error loading data: {e}")
                self.queue.put(None)
                break

    def __iter__(self):
        if not self.started:
            self.start()
        return self

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

class DataPrefetchLoader:

    def __init__(self, dataloader, device, queue_size=2):
        self.prefetch_generator = DataPrefetchGenerator(dataloader, device, queue_size)

    def __iter__(self):
        return iter(self.prefetch_generator)

    def __len__(self):
        return len(self.prefetch_generator.dataloader)


