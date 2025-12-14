import torch
import numpy
import argparse
import yaml
import os

from PyQt5.QtCore import QLibraryInfo
from imutils import paths
from torch.nn import BCEWithLogitsLoss

from train import Trainer
from test import Tester
from utils.dataloader import Dataloader
from utils.plot import PlotModel
from utils.show_results import Shower
from model.hourglass import HourglassNet
from model.unet import Unet


def main(args):

    os.makedirs("./output", exist_ok= True)

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.gpu:
        with open('config_gpu.yaml', 'r') as file:
            config_file = yaml.safe_load(file) 
            imagePaths = sorted(list(paths.list_images(config_file["dataset"]["images_path"])))
            maskPaths = sorted(list(paths.list_images(config_file["dataset"]["masks_path"])))

    else:
        with open('config.yaml', 'r') as file:
            config_file = yaml.safe_load(file) 
            imagePaths = sorted(list(paths.list_images(config_file["dataset"]["images_path"])))
            maskPaths = sorted(list(paths.list_images(config_file["dataset"]["masks_path"])))
        
    data_loader = Dataloader(image_path= imagePaths, mask_path= maskPaths, config_file= config_file)
    train_loader, val_loader, test_loader = data_loader.get_loader() 

    print(len(train_loader), len(val_loader), len(test_loader))

    if args.train:
        trainer = Trainer(config= config_file, train_loader= train_loader, val_loader= val_loader, device= device)

        if args.hourglass:
            os.makedirs("./output/checkpoints/hourglass/", exist_ok= True)
            model = HourglassNet(in_channels= 3, hidden_dim = [8, 16, 32, 64], n_channels = 1)
            model.to(device= device)
            trainer.train_hourglass(model= model)
        
        if args.unet:
            os.makedirs("./output/checkpoints/unet/", exist_ok= True)
            model = Unet(in_channels= 3, hidden_dim= [8, 16, 32, 64], n_channels= 1)
            model.to(device= device)
            trainer.train_unet(model= model)

    if args.test:
        tester = Tester(config= config_file, device= device)

        if args.hourglass:
            model = HourglassNet(in_channels= 3, hidden_dim = [8, 16, 32, 64], n_channels = 1)
            Loss = BCEWithLogitsLoss()
            plot = PlotModel(model= model, device= device, in_channel= 3, img_size= config_file["input"]["height"], path= "./output/checkpoints/hourglass")
            
            plot.plot_model(input= plot.input_model())
            
            state_dict = torch.load("./output/checkpoints/hourglass/final_model.pth", map_location=device)
            model.load_state_dict(state_dict)
            model, avg_loss = tester.test_model(model= model, test_loader= test_loader, Loss= Loss)
            
            predictor = Shower(model=model, device=device, test_loader=test_loader, output_dir= "./output/figures/hourglass/")
            predictor.predict_test_set()

        if args.unet:
            model = Unet(in_channels= 3, hidden_dim= [8, 16, 32, 64], n_channels= 1)
            Loss = BCEWithLogitsLoss()
            plot = PlotModel(model= model, device= device, in_channel= 3, img_size= config_file["input"]["height"], path= "./output/checkpoints/unet")
            
            plot.plot_model(input= plot.input_model())
            
            state_dict = torch.load("./output/checkpoints/unet/final_model.pth", map_location=device)
            model.load_state_dict(state_dict)
            model, avg_loss = tester.test_model(model= model, test_loader= test_loader, Loss= Loss)
            
            predictor = Shower(model=model, device=device, test_loader=test_loader, output_dir= "./output/figures/unet/")
            predictor.predict_test_set()




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
