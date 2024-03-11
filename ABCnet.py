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

encoder = nn.Sequential(
            nn.LayerNorm([256, 64]),
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
            nn.Linear(64, 2),
        )

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.branch3x3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(nn.functional.max_pool1d(x, kernel_size=3, stride=1, padding=1))
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        outputs = [branch5x5]
        
        outputs = torch.cat(outputs, 1)  # Concatenate along the channel dimension
        return outputs

class RadarGestureNet(L.LightningModule):
    def __init__(self, gesture_class):
        super().__init__()
        self.gesture_class = gesture_class
        self.save_hyperparameters()
        
        self.Icp1 = nn.Sequential(
            nn.LayerNorm([30, 64]),
            InceptionModule(30, 64)
        )
        
        self.Icp2 = nn.Sequential(
            nn.LayerNorm([30, 64]),
            InceptionModule(30, 64)
        )
        self.Icp3 = nn.Sequential(
            nn.LayerNorm([30, 64]),
            InceptionModule(30, 64)
        )
        
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x1, x2, x3):
        embedding = self.Icp1(x1)+self.Icp2(x2)+self.Icp3(x3)
        
        embedding = self.decoder(embedding)
        return embedding
    def training_step(self, batch, batch_idx):
        x1,x2,x3, y = batch
        z = self.forward(x1, x2, x3)
        criterion = nn.MSELoss()
        loss = criterion(z, y)
        self.log("train_loss", loss)
        train_accuracy = torch.sum(one_hot_to_label(z) == one_hot_to_label(y)).item() / len(y)
        self.log("train_accuracy", train_accuracy)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x1,x2,x3, y = batch
        z = self.forward(x1, x2, x3)
        criterion = nn.MSELoss()
        val_loss = criterion(z, y)
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx):
        x1,x2,x3, y = batch
        z = self.forward(x1, x2, x3)
        criterion = nn.MSELoss()
        
        test_loss = criterion(z, y)
        self.log("test_loss", test_loss)
        
        accuracy = torch.sum(one_hot_to_label(z) == one_hot_to_label(y)).item() / len(y)
        self.log("accuracy", accuracy)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer