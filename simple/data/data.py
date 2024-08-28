import os
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = os.listdir(image_folder)

    def __getitem__(self, idx):
        image_file = self.images[idx]

    def __len__(self):
        return len(self.images)
