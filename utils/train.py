import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

class Trainer():
    def __init__(self, config):
        self.config = config
        
        
    def train(self, model):

        self.lr = self.config["training"]["lr"]
        self.epochs = self.config["training"]["epochs"]
        self.batch_size = self.config["training"]["batch_size"]
        self.threshold = self.config["training"]["threshold"]
        self.model = model
        self.model.train()

 