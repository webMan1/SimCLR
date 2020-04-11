import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from models.resnet_simclr import ResNetSimCLR
from data_aug.stanfordcars import CarsDataset
from data_aug.compcars import CompCars

def encode(save_root, model_file, data_folder, model_name='ca', dataset_name='celeba', batch_size=64, device='cuda:0', out_dim=256):
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)

    if dataset_name == 'celeba':
        train_loader = DataLoader(datasets.CelebA(data_folder, split='train', download=True, transform=transforms.ToTensor()),
                                    batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(datasets.CelebA(data_folder, split='valid', download=True, transform=transforms.ToTensor()),
                                    batch_size=batch_size, shuffle=False)
    elif dataset_name == 'stanfordCars':
        t = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x)
        ])
        train_data_dir = os.path.join(data_folder, 'cars_train/')
        train_annos = os.path.join(data_folder, 'devkit/cars_train_annos.mat')
        train_loader = DataLoader(CarsDataset(train_annos, train_data_dir, t), batch_size=batch_size, shuffle=False)
        valid_data_dir = os.path.join(data_folder, 'cars_test/')
        valid_annos = os.path.join(data_folder, 'devkit/cars_test_annos_withlabels.mat')
        valid_loader = DataLoader(CarsDataset(valid_annos, valid_data_dir, t), batch_size=batch_size, shuffle=False)
    elif dataset_name == 'compCars':
        t = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])
        train_loader = DataLoader(CompCars(data_folder, True, t), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(CompCars(data_folder, False, t), batch_size=batch_size, shuffle=False)


    model = ResNetSimCLR('resnet50', out_dim)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()

    print('Starting on training data')
    train_encodings = []
    for x, _ in train_loader:
        x = x.to(device)
        h, _ = model(x)
        train_encodings.append(h.cpu().detach())
    torch.save(torch.cat(train_encodings, dim=0), os.path.join(save_root, f'{dataset_name}-{model_name}model-train_encodings.pt'))

    print('Starting on validation data')
    valid_encodings = []
    for x, _ in valid_loader:
        x = x.to(device)
        h, _ = model(x)
        if len(h.shape) == 1:
            h = h.unsqueeze(0)
        valid_encodings.append(h.cpu().detach())
    torch.save(torch.cat(valid_encodings, dim=0), os.path.join(save_root, f'{dataset_name}-{model_name}model-valid_encodings.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--save_root', type=str, default='./encodings')
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, default='celeba')
    parser.add_argument('--model_name', type=str, default='ca')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--out_dim', type=int, default=256)

    config = parser.parse_args()
    encode(config.save_root, config.model_file, config.data_folder, config.model_name, 
            config.dataset_name, config.batch_size, config.device, config.out_dim)
