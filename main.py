import torch
import numpy
import argparse
import yaml
import os

from PyQt5.QtCore import QLibraryInfo
from imutils import paths

from utils.dataloader import Dataloader
from utils.train import Trainer
from model.hourglass import HourglassNet


def main(args):

    os.makedirs("./output", exist_ok= True)
    os.makedirs("./figures", exist_ok= True)

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.gpu:
        with open('utils/config_gpu.yaml', 'r') as file:
            config_file = yaml.safe_load(file) 
            imagePaths = sorted(list(paths.list_images(config_file["dataset"]["images_path"])))
            maskPaths = sorted(list(paths.list_images(config_file["dataset"]["masks_path"])))

    else:
        with open('utils/config.yaml', 'r') as file:
            config_file = yaml.safe_load(file) 
            imagePaths = sorted(list(paths.list_images(config_file["dataset"]["images_path"])))
            maskPaths = sorted(list(paths.list_images(config_file["dataset"]["masks_path"])))
        
    data_loader = Dataloader(image_path= imagePaths, mask_path= maskPaths, config_file= config_file)
    train_loader, test_loader = data_loader.get_loader() 

    if args.train:
        trainer = Trainer(config= config_file)

        if args.hourglass:
            os.makedirs("./figures/hourglass/", exist_ok= True)
            model = HourglassNet(in_channels= 3, hidden_dim = [8, 16, 32, 64], n_channels = 2)

            trainer.train(model= model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help= "Run on the gpu cluster")
    parser.add_argument("--train", action="store_true", help="training mode")
    parser.add_argument("--test", action="store_true", help="testing mode")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hourglass", action="store_true", help="Run hourglass model")
    group.add_argument("--unet", action="store_true", help="Run unet model")

    args = parser.parse_args()
    main(args)