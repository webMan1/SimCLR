import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from models.resnet_simclr import ResNetSimCLR

def encode(save_root, model_file, data_folder, dataset_name='celeba', batch_size=64, device='cuda:0', out_dim=256):
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)

    if dataset_name == 'celeba':
        train_loader = DataLoader(datasets.CelebA(data_folder, split='train', download=True),
                                    batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(datasets.CelebA(data_folder, split='valid', download=True),
                                    batch_size=batch_size, shuffle=False)

    model = ResNetSimCLR('resnet50', out_dim)
    model.load_state_dict(torch.load(model_file), map_location=device)
    model.eval()

    print('Starting on training data')
    train_encodings = []
    for x, _ in train_loader:
        x = x.to(device)
        h, _ = model(x)
        train_encodings.append(h.cpu().detach())
    torch.save(torch.cat(train_encodings, dim=0), os.path.join(save_root, 'train_encodings.pt'))

    print('Starting on validation data')
    valid_encodings = []
    for x, _ in valid_loader:
        x = x.to(device)
        h, _ = model(x)
        valid_encodings.append(h.cpu().detach())
    torch.save(torch.cat(valid_encodings, dim=0), os.path.join(save_root, 'valid_encodings.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--save_root', type=str, default='./encodings')
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, default='celeba')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--out_dim', type=int, default=256)

    config = parser.parse_args()
    encode(config.save_root, config.model_file, config.data_folder, 
            config.dataset_name, config.batch_size, config.device, config.out_dim)