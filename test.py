from tqdm import tqdm

class Tester():
    """
        Utility class for evaluating a trained model on a test dataset.

        This class runs the model in evaluation mode, computes the test loss,
        and reports progress using a progress bar.
    """
    def __init__(self, config, device):
        """
            Initialize the Tester.

            Args:
                config (dict): Configuration dictionary containing evaluation settings.
                device (torch.device): Device on which the model and data are evaluated
                    (e.g., CPU or CUDA).
        """
        self.config = config
        self.device = device

    def test_model(self, model, test_loader, Loss):
        """
            Evaluate the model on the test dataset.

            The model is set to evaluation mode and the loss is computed for each
            batch in the test set. A progress bar displays the current batch loss.

            Args:
                model (torch.nn.Module): Trained model to be evaluated.
                test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
                Loss (callable): Loss function used to compute the test loss.

            Returns:
                tuple:
                    - model (torch.nn.Module): The evaluated model.
                    - avg_loss (float): Average test loss over the entire test dataset.
        """
        
        model.to(self.device)
        model.eval()

        batch_bar = tqdm(test_loader, desc="Test Loss: 0.000")

        for step, (img_batch, label_batch) in enumerate(batch_bar):

            img_batch = img_batch.to(self.device)
            label_batch = label_batch.to(self.device)

            prediction = model(img_batch)
            loss = Loss(prediction, label_batch)
            batch_bar.set_description(f"Test Loss: {loss.item():.4f}")

        avg_loss = sum(loss.item() * img_batch.size(0) for img_batch, label_batch in test_loader) / len(test_loader)
        print(f"Test loss: {avg_loss:.2f}")

        return model, avg_loss
    
