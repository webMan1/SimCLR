import os
import torch
from torch.utils.data import Dataset
import scipy.io
from PIL import Image

class CarsDataset(Dataset):
    def __init__(self, mat_anno, data_dir, transform=None):
        self.full_data_set = scipy.io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name)
        car_class = self.car_annotations[idx][-2][0][0] -1

        if self.transform:
            image = self.transform(image)
        y = torch.LongTensor([car_class])
        return image, y

class StanfordCarsMini(Dataset):
    def __init__(self, mat_anno, data_dir, transform=None):
        average_cars_file = os.path.join(os.path.dirname(__file__), 'most_average_stanford_cars.mod')
        self.average_cars = torch.load(average_cars_file)['averages']
        self.full_dataset = CarsDataset(mat_anno, data_dir, transform)

    def __len__(self):
        return len(self.average_cars)

    def __getitem__(self, idx):
        real_idx = self.average_cars[idx]['index']
        return self.full_dataset[real_idx]
