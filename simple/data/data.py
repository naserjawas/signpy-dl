import os
import torch

from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, list_ids, img_dir, labels):
        self.list_ids = list_ids
        self.img_dir = img_dir
        self.labels = labels

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        pass
