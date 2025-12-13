import torch

from tqdm import tqdm

class Tester():
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def test_hourglass(self, model, test_loader, Loss):
        
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
