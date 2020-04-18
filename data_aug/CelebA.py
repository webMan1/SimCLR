import torch
from torch.utils.data import Dataset
from torchvision import datasets

class SubCelebaA(Dataset):
    def __init__(self, data_root, is_train:bool, selected_attribute:str, exclude:bool=True, transform=None):
        super(Dataset, self).__init__()

        dataset = datasets.CelebA(data_root, split='train' if is_train else 'valid', download=True,
                                    transform=transform)

        remove_me = dataset.attr_names.index(selected_attribute)
        mask = torch.zeros_like(dataset.attr[0])
        mask[remove_me] = 1

        if exclude:
            self.selected_indices = (torch.any((dataset.attr & mask).type(torch.uint8), dim=1) == 0).nonzero()
        else:
            self.selected_indices =  torch.any((dataset.attr & mask).type(torch.uint8), dim=1).nonzero()

        self.dataset = dataset
    
    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        translated_index = self.selected_indices[idx]
        return self.dataset[translated_index]
