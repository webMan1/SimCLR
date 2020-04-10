import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CompCars(Dataset):
    def __init__(self, data_root, train=True, transform=None):
        self.transform = transform
        self.data_root = data_root
        if train:
            split_file = os.path.join(data_root, 'train_test_split/classification/train.txt')
        else:
            split_file = os.path.join(data_root, 'train_test_split/classification/test.txt')
        self.images = []
        with open(split_file, 'r') as f:
            for line in f:
                self.images.append(line.strip())

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        filename = os.path.join(self.data_root, 'image/', self.images[idx])
        img = Image.open(filename)
        if self.transform:
            img = self.transform(img)

        return img, torch.LongTensor([1])
