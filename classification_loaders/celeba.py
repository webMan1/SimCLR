import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CelebADataset(Dataset):
    '''
    one of target_col_idx or target should be supplied:
        target_col_idx specifies the column index from the attributes to use as the label
        target specifies a target that spans multiple columns
    '''
    def __init__(self, data_root, split='train', target_col_idx=None, target=None):
        split_map = {
            "train": 0,
            "valid": 1,
        }
        assert split in split_map
        assert (target_col_idx or target) and not (target_col_idx and target)
        if target:
            assert target in ['hair']

        splits = pd.read_csv(os.path.join(data_root, "list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pd.read_csv(os.path.join(data_root, "list_attr_celeba.txt"), delim_whitespace=True, header=1)
        mask = splits[1] == split_map[split]

        if target_col_idx:
            print(f'Using the {attr.columns[target_col_idx]} attribute as the label')
            self.labels = torch.as_tensor(attr[mask].values[:, target_col_idx])
            self.labels = self.labels.reshape(-1, 1)
        else:
            # TODO: probably should do some argmax or something
            print(f'Using a group of columns to represent {target}')
            if target == 'hair':
                # bald, black_hair, blonde_hair, brown_hair, gray_hair, receding_hairline
                col_idxs = [4, 8, 9, 11, 17, 28]
            self.labels = torch.as_tensor(attr[mask].values[:, col_idxs])
        self.labels = (self.labels + 1) // 2 # changes it from -1,1 to 0,1
        print(self.labels.shape)
            
        self.encodings = torch.load(os.path.join(data_root, f'{split}_encodings.pt'))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        return self.encodings[i], self.labels[i]