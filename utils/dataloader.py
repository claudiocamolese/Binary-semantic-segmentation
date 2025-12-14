import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import SegmentationDataset
import numpy as np
from PIL import Image

class Dataloader():
    """
        Utility class to create PyTorch DataLoaders for image segmentation tasks.

        This class:
        - Splits the dataset into train, validation, and test sets
        - Applies image and mask transformations
        - Optionally computes mean and standard deviation on the training set
        - Exposes PyTorch DataLoader objects for each split
    """
    def __init__(self, image_path, mask_path, config_file):
        """
            Initialize the Dataloader.

            Args:
                image_path (list[str]): List of file paths to input images.
                mask_path (list[str]): List of file paths to corresponding segmentation masks.
                config_file (dict): Configuration dictionary containing:
                    - split ratios (train/val/test)
                    - input image size
                    - training parameters (e.g., batch size)
        """
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
            test_size=self.config["split"].get("val", 0.1),
            random_state=42
        )

        # calcolo dinamico mean e std sul train set
        #train_mean, train_std = self.compute_train_mean_std(trainImages)

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
            transforms.ToTensor()
        ])

        # dataset
        train_set = SegmentationDataset(trainImages, trainMasks, image_transform, mask_transform)
        val_set = SegmentationDataset(valImages, valMasks, image_transform, mask_transform)
        test_set = SegmentationDataset(testImages, testMasks, image_transform, mask_transform)

        batch_size = self.config["training"]["hourglass"]["batch_size"]
        pin_memory = torch.cuda.is_available()
        num_workers = os.cpu_count()

        # DataLoader
        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
        self.val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)
        self.test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
        
        self.testImages = testImages

    def compute_train_mean_std(self, image_paths):
        """
            Compute per-channel mean and standard deviation on the training images.

            The statistics are computed in RGB space and normalized in the [0, 1] range.

            Args:
                image_paths (list[str]): List of file paths to training images.

            Returns:
                tuple[list[float], list[float]]:
                    - mean: Per-channel RGB mean
                    - std: Per-channel RGB standard deviation
        """
        mean = np.zeros(3)
        std = np.zeros(3)
        for path in image_paths:
            img = np.array(Image.open(path).convert('RGB')) / 255.0  # normalizza tra 0 e 1
            mean += img.mean(axis=(0, 1))
            std += img.std(axis=(0, 1))
        mean /= len(image_paths)
        std /= len(image_paths)
        return mean.tolist(), std.tolist()

    def get_loader(self):
        """
            Return the train, validation, and test DataLoaders.

            Returns:
                tuple[DataLoader, DataLoader, DataLoader]:
                    Train, validation, and test DataLoader objects.
        """
        return self.train_loader, self.val_loader, self.test_loader
