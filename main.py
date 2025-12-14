import torch
import argparse
import yaml
import os

from imutils import paths
from torch.nn import BCEWithLogitsLoss

from train import Trainer
from test import Tester
from utils.dataloader import Dataloader
from utils.plot import PlotModel
from utils.printer import printing_model
from utils.show_results import Shower
from model.hourglass import HourglassNet
from model.unet import Unet


def main(args):

    os.makedirs("./output", exist_ok= True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
            printing_model(model= model, model_name= "hourglass")
            trainer.train_hourglass(model= model)
        
        if args.unet:
            if args.big:
                 os.makedirs("./output/checkpoints/big_unet/", exist_ok= True)
                 model_name = "big_unet"
            else:
                os.makedirs("./output/checkpoints/unet/", exist_ok= True)
                model_name = "unet"

            model = Unet(in_channels= 3, hidden_dim= [32, 62, 128, 256] if args.big else [8, 16, 32, 64], n_channels= 1)
            model.to(device= device)
            printing_model(model= model, model_name= model_name)
            trainer.train_unet(model= model, model_name= model_name)

    if args.test:
        tester = Tester(config= config_file, device= device)

        if args.hourglass:
            model = HourglassNet(in_channels= 3, hidden_dim = [8, 16, 32, 64], n_channels = 1)
            printing_model(model= model, model_name= "hourglass")
            Loss = BCEWithLogitsLoss()
            plot = PlotModel(model= model, device= device, in_channel= 3, img_size= config_file["input"]["height"], path= "./output/checkpoints/hourglass")
            
            plot.plot_model(input= plot.input_model())
            
            state_dict = torch.load("./output/checkpoints/hourglass/final_model.pth", map_location=device)
            model.load_state_dict(state_dict)
            model, avg_loss = tester.test_model(model= model, test_loader= test_loader, Loss= Loss)
            
            predictor = Shower(model=model, device=device, test_loader=test_loader, output_dir= "./output/figures/hourglass/")
            predictor.predict_test_set()

        if args.unet:
            model = Unet(in_channels= 3, hidden_dim= [32, 62, 128, 256] if args.big else [8, 16, 32, 64], n_channels= 1)
            Loss = BCEWithLogitsLoss()
            if args.big:
                model_name = "big_unet"
            else:
                model_name = "unet"
            
            printing_model(model= model, model_name= model_name)
            plot = PlotModel(model= model, device= device, in_channel= 3, img_size= config_file["input"]["height"], path= f"./output/checkpoints/{model_name}")
            
            plot.plot_model(input= plot.input_model())
            
            state_dict = torch.load(f"./output/checkpoints/{model_name}/final_model.pth", map_location=device)
            model.load_state_dict(state_dict)
            model, avg_loss = tester.test_model(model= model, test_loader= test_loader, Loss= Loss)
            
            predictor = Shower(model=model, device=device, test_loader=test_loader, output_dir= f"./output/figures/{model_name}/")
            predictor.predict_test_set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="training mode")
    parser.add_argument("--test", action="store_true", help="testing mode")
    parser.add_argument("--big", action="store_true", help= "Run bigger model")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hourglass", action="store_true", help="Run hourglass model")
    group.add_argument("--unet", action="store_true", help="Run unet model")

    args = parser.parse_args()
    main(args)
