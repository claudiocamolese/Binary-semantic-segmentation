import torch
import os

from sklearn.model_selection import train_test_split
from torchvision import transforms

from .dataset import SegmentationDataset

from torch.utils.data import DataLoader


class Dataloader():
    def __init__(self, image_path, mask_path, config_file):
        self.imagePaths = image_path
        self.maskPaths = mask_path
        self.config_file = config_file
        self.test_size = self.config_file["split"]["test"]

        split = train_test_split(self.imagePaths, self.maskPaths,
                             test_size=self.config_file["split"]["test"], random_state=42)

        # unpack the data split
        (trainImages, testImages) = split[:2]
        (trainMasks, testMasks) = split[2:]
        
        train_set = SegmentationDataset(imagePaths= trainImages, maskPaths= trainMasks, 
                                        transforms= transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((self.config_file["input"]["height"], self.config_file["input"]["width"])),
                                            transforms.ToTensor()
                                        ]))
        test_set = SegmentationDataset(imagePaths= testImages, maskPaths= testMasks, 
                                        transforms= transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((self.config_file["input"]["height"], self.config_file["input"]["width"])),
                                            transforms.ToTensor()
                                        ]))

        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=self.config_file["training"]["batch_size"], pin_memory=True if torch.cuda.is_available() else False, num_workers=os.cpu_count())
        self.test_loader = DataLoader(test_set, shuffle=False, batch_size=self.config_file["training"]["batch_size"], pin_memory=True if torch.cuda.is_available() else False, num_workers=os.cpu_count())

    def get_loader(self):
        return self.train_loader, self.test_loader


