import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from data_aug.stanfordcars import StanfordCarsMini, CarsDataset
from models.resnet_simclr import ResNetSimCLR
from solver import CESolver

class FineTuner(nn.Module):
    def __init__(self, embedder, clf):
        super(FineTuner, self).__init__()
        self.embedder = embedder
        self.clf = clf

    def forward(self, x):
        h, _ = self.embedder(x)
        y_hat = self.clf(h)
        return y_hat

def run(config):
    model = ResNetSimCLR('resnet50', config.out_dim)
    model.load_state_dict(torch.load(config.model_file, map_location=config.device))
    model = model.to(config.device)
    clf = nn.Linear(2048, 196)
    full = FineTuner(model, clf)
    optim = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()))
    objective = nn.CrossEntropyLoss()

    t = T.Compose([
            T.Resize(512),
            T.CenterCrop(512),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0] == 1 else x)
        ])
    train_data_dir = os.path.join(config.data_root, 'cars_train/')
    train_annos = os.path.join(config.data_root, 'devkit/cars_train_annos.mat')
    train_dataset = StanfordCarsMini(train_annos, train_data_dir, t)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    valid_data_dir = os.path.join(config.data_root, 'cars_test/')
    valid_annos = os.path.join(config.data_root, 'devkit/cars_test_annos_withlabels.mat')
    valid_dataset = CarsDataset(valid_annos, valid_data_dir, t)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

    solver = CESolver(full, train_loader, valid_loader, config.save_root, name=config.name, device=config.device)
    solver.train(config.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help='location of model to fine tune')
    parser.add_argument('--data_root', type=str, default='/multiview/datasets/StanfordCars')
    parser.add_argument('--save_root', type=str, default='./finetuned', help='location to save the fine tuned model')
    parser.add_argument('--num_epochs', type=int, default=80, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_dim', type=int, default=256, help='dimension of projection')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--name', type=str, default='SCMTuned')

    args = parser.parse_args()
    run(args)