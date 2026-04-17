import os
import bisect
import logging
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir:str):
        self.data_dir = data_dir
        self.index_map =
        