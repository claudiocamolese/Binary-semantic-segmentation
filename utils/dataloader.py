import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import SegmentationDataset

class Dataloader():
    def __init__(self, image_path, mask_path, config_file):
        self.imagePaths = image_path
        self.maskPaths = mask_path
        self.config = config_file

        # split train/test
        trainImages, testImages, trainMasks, testMasks = train_test_split(
            self.imagePaths, self.maskPaths,
            test_size=self.config["split"]["test"], random_state=42
        )

        # split train/val
        trainImages, valImages, trainMasks, valMasks = train_test_split(
            trainImages, trainMasks,
            test_size=self.config["split"].get("val", 0.1),  # default 10% val
            random_state=42
        )

        # media e std RGB (puoi calcolare dinamicamente se vuoi)
        train_mean = [0.5203, 0.4543, 0.4254]
        train_std = [0.2525, 0.2355, 0.2301]

        # trasformazioni
        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config["input"]["height"], self.config["input"]["width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=train_mean, std=train_std)
        ])

        mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config["input"]["height"], self.config["input"]["width"])),
            transforms.ToTensor()  # senza Normalize
        ])

        # dataset
        train_set = SegmentationDataset(trainImages, trainMasks, image_transform, mask_transform)
        val_set = SegmentationDataset(valImages, valMasks, image_transform, mask_transform)
        test_set = SegmentationDataset(testImages, testMasks, image_transform, mask_transform)

        batch_size = self.config["training"]["hourglass"]["batch_size"]
        pin_memory = torch.cuda.is_available()
        num_workers = os.cpu_count()

        # DataLoader
        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size,
                                       pin_memory=pin_memory, num_workers=num_workers)
        self.val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size,
                                     pin_memory=pin_memory, num_workers=num_workers)
        self.test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size,
                                      pin_memory=pin_memory, num_workers=num_workers)
        
        self.testImages = testImages

    def get_loader(self):
        return self.train_loader, self.val_loader, self.test_loader