from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
    """
        PyTorch Dataset for image segmentation tasks.

        This dataset loads RGB images and their corresponding segmentation masks,
        applies optional transformations, and returns image mask pairs suitable
        for training, validation, or testing segmentation models.
    """
    def __init__(self, imagePaths, maskPaths, image_transforms=None, mask_transforms=None):
        """
            Initialize the SegmentationDataset.

            Args:
                imagePaths (list[str]): List of file paths to input RGB images.
                maskPaths (list[str]): List of file paths to corresponding segmentation masks.
                image_transforms (callable, optional): Transformations applied to the input images.
                    Typically includes resizing, normalization, and conversion to tensor.
                mask_transforms (callable, optional): Transformations applied to the segmentation masks.
                    Typically includes resizing and conversion to tensor.
        """
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        """
            Return the total number of samples in the dataset.

            Returns:
                int: Number of image mask pairs.
        """
        return len(self.imagePaths)

    def __getitem__(self, idx):
        """
            Retrieve the image and mask corresponding to the given index.

            Args:
                idx (int): Index of the sample to retrieve.

            Returns:
                tuple: A tuple (image, mask) where:
                    - image (Tensor): Transformed RGB image.
                    - mask (Tensor): Transformed segmentation mask.
        """
        image = cv2.imread(self.imagePaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[idx], 0)  # grayscale

        if self.image_transforms:
            image = self.image_transforms(image)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        return image, mask
