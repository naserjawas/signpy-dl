import torch

from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, list_ids, labels):
        self.labels = labels
        self.list_ids = list_ids

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        pass
