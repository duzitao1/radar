import torch
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import lightning as L

def one_hot_labels(caategorical_labels):
    enc = OneHotEncoder(handle_unknown='ignore')
    on_hot_labels = enc.fit_transform(
        caategorical_labels.reshape(-1, 1)).toarray()
    return on_hot_labels
def one_hot_to_label(one_hot):
    return torch.argmax(one_hot, dim=1)

class RadarGestureNet(L.LightningModule):
    def __init__(self, encoder=None, gesture_class=2):
        super().__init__()
        self.gesture_class = gesture_class
        self.save_hyperparameters()
        
        if encoder==None:
            self.encoder = nn.Sequential(
            nn.LayerNorm([30, 64]),
            nn.Conv1d(30, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, gesture_class),
        )
        
        else:
            self.encoder = encoder
        
        
        
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        criterion = nn.MSELoss()
        loss = criterion(z, y)
        self.log("train_loss", loss)
        train_accuracy = torch.sum(one_hot_to_label(z) == one_hot_to_label(y)).item() / len(y)
        self.log("train_accuracy", train_accuracy)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        criterion = nn.MSELoss()
        val_loss = criterion(z, y)
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        criterion = nn.MSELoss()
        
        test_loss = criterion(z, y)
        self.log("test_loss", test_loss)
        
        accuracy = torch.sum(one_hot_to_label(z) == one_hot_to_label(y)).item() / len(y)
        self.log("accuracy", accuracy)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
