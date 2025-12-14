import torch
import numpy as np
import matplotlib.pyplot as plt
import os


class Shower:
    """
        Utility class for visualizing model predictions on the test set.

        The class runs inference on images from the test DataLoader and saves
        comparison plots (input image, ground-truth mask, predicted mask, and
        predicted probability map) to the specified output directory.
    """
    def __init__(self, model, device, test_loader, output_dir,num_images=20):
        """
            Initialize the Shower.

            Args:
                model (torch.nn.Module): Trained PyTorch model.
                device (torch.device or str): Device used for inference ("cuda" or "cpu").
                test_loader (torch.utils.data.DataLoader): DataLoader containing test images.
                output_dir (str): Directory where prediction plots will be saved.
                num_images (int, optional): Number of test images to visualize.
                    Defaults to 20.
        """
        self.model = model.to(device)
        self.device = device
        self.test_loader = test_loader
        self.model.eval()
        self.num_images = num_images
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)

        # valori di media e std per denormalizzazione
        self.train_mean = np.array([0.5203, 0.4543, 0.4254])
        self.train_std = np.array([0.2525, 0.2355, 0.2301])

    @staticmethod
    def _get_img_name(imagePath):
        """
            Extract the image name from a file path (without extension).

            Args:
                imagePath (str): Full path to the image file.

            Returns:
                str: Image file name without extension.
        """
        imagePathR = "".join(reversed(imagePath))
        pos = imagePathR.find('/')
        return imagePath[len(imagePath)-pos:len(imagePath)-4]

    def _denormalize(self, img):
        """
            Denormalize a normalized image.

            Args:
                img (np.ndarray): Image in HWC format with values in [0, 1].

            Returns:
                np.ndarray: Denormalized image with values clipped to [0, 1].
        """
        img = img * self.train_std + self.train_mean
        img = np.clip(img, 0, 1)
        return img

    def _prepare_plot(self, origImage, origMask, predMask, predProb, imageName):
        """
            Create and save a visualization plot for a single prediction.

            The plot includes:
            - Original RGB image
            - Ground-truth mask
            - Binary predicted mask
            - Predicted probability map

            Args:
                origImage (np.ndarray): Original RGB image (HWC).
                origMask (np.ndarray): Ground-truth mask.
                predMask (np.ndarray): Binary predicted mask.
                predProb (np.ndarray): Predicted probability map.
                imageName (str): Name used to save the plot.
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()

        axs[0].imshow(origImage)
        axs[1].imshow(np.squeeze(origMask), cmap='gray')
        axs[2].imshow(np.squeeze(predMask), cmap='gray')
        axs[3].imshow(np.squeeze(predProb)/255.0, cmap='gray')

        axs[0].set_title("Image")
        axs[1].set_title("Original Mask")
        axs[2].set_title("Predicted Mask")
        axs[3].set_title("Predicted Probability")

        fig.tight_layout()

        plot_path = os.path.join(self.output_dir, f"predict_plot_{imageName}.png")
        fig.savefig(plot_path)
        plt.close(fig)

    def predict_test_set(self):
        """
            Run inference on the first N images of the test set and save prediction plots.

            The method iterates over the test DataLoader, performs forward passes,
            and generates visualization plots until the specified number of images
            is reached.
        """
        self.model.eval()
        images_count = 0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.test_loader):
                for i in range(images.size(0)):
                    if images_count >= self.num_images:
                        return  # fermati dopo le prime N immagini

                    img_tensor = images[i].cpu().numpy().transpose(1, 2, 0)  # C,H,W → H,W,C
                    img_tensor = self._denormalize(img_tensor)  # denormalizzazione
                    origMask = masks[i].cpu().numpy()

                    input_tensor = images[i].unsqueeze(0).to(self.device)
                    predMask = self.model(input_tensor).squeeze()
                    predMask = torch.sigmoid(predMask).cpu().numpy()

                    predProb = predMask * 255
                    predMask_bin = (predMask > 0.5) * 255
                    predMask_bin = predMask_bin.astype(np.uint8)

                    imageName = f"img{images_count}"
                    self._prepare_plot(img_tensor, origMask, predMask_bin, predProb, imageName)

                    images_count += 1
