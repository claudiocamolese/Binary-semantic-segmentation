import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.nn import BCEWithLogitsLoss

from test import Tester

class Trainer():
    def __init__(self, config, train_loader, val_loader, device):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.tester = Tester(config= self.config, device= self.device)
        
    def train_hourglass(self, model):

        self.lr = self.config["training"]["hourglass"]["lr"]
        self.epochs = self.config["training"]["hourglass"]["epochs"]


        self.model = model
        self.model.train()

        loss = BCEWithLogitsLoss()

        model_name ="hourglass"

        self.train(model= self.model, Loss = loss, lr= self.lr, epochs= self.epochs, model_name= model_name)

    def train_unet(self, model):

        self.lr = self.config["training"]["unet"]["lr"]
        self.epochs = self.config["training"]["unet"]["epochs"]

        self.model = model
        self.model.train()

        loss = BCEWithLogitsLoss()

        model_name ="unet"

        self.train(model= self.model, Loss = loss, lr= self.lr, epochs= self.epochs, model_name= model_name)

    def train(self, model, Loss, lr, epochs, model_name):

        last_loss = float('inf')

        optimizer = Adam(model.parameters(), lr= lr)
        scheduler = CosineAnnealingLR(optimizer= optimizer, T_max= epochs, eta_min=1e-5)

        for epoch in range(epochs):
            avg_loss = 0.
            num_items = 0

            batch_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Loss: 0.000")

            for step, (img_batch, gt_batch) in enumerate(batch_bar):
                img_batch = img_batch.to(self.device)
                gt_batch = gt_batch.to(self.device)

                prediction = model(img_batch)
                loss = Loss(prediction, gt_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() * img_batch.shape[0]
                num_items += img_batch.shape[0]
                
                batch_bar.set_description(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")
                   
            scheduler.step()
            lr_current = scheduler.get_last_lr()[0]

            epoch_loss = avg_loss / num_items
            print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch + 1, epoch_loss, lr_current))
            
            model, val_loss =  self.tester.test_model(model= self.model, test_loader= self.val_loader, Loss= Loss)
            model.train()

            if val_loss < last_loss:
                torch.save(model.state_dict(), f"./output/checkpoints/{model_name}/final_model.pth")
                last_loss = val_loss
                print(f"New model saved in output/checkpoints/{model_name}/ !")
                



