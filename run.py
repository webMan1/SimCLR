from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import torch

losses = []

def get_valid_callback(save_loc:str):
    if save_loc is None:
        return lambda m,e,l: None
    else:    
        def each_valid(model, epoch, valid_loss):
            losses.append(valid_loss)

            state = {
                "model": model.state_dict(),
                "epoch": epoch, 
                "valid_losses": losses
            }
            torch.save(state, save_loc)
        return each_valid

def main(save_loc:str=None):
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)

    callback = get_valid_callback(save_loc)

    simclr.train(callback)


if __name__ == "__main__":
    main()
