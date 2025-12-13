from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, image_transforms=None, mask_transforms=None):
        """
        image_transforms: trasformazioni per immagini RGB
        mask_transforms: trasformazioni per mask grayscale
        """
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        image = cv2.imread(self.imagePaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[idx], 0)  # grayscale

        if self.image_transforms:
            image = self.image_transforms(image)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        return image, mask
