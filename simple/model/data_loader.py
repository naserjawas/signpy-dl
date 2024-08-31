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

"""
Notes dataset

the dataset contains:
    - train:
    - dev:
    - test:

    each data folder has:
    - label
    - folder of sample.


    train/word1/sample1/
    train/word1/sample2/
    train/word1/sample3/
    train/word2/sample1/
    train/word2/sample2/
    train/word2/sample3/
"""
