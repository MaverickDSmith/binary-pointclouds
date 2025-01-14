import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

# Feature Transformation Network
class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(self.k, device=x.device).repeat(batch_size, 1, 1)
        x = x.view(batch_size, self.k, self.k) + identity
        return x


# PointNet Model
class PointNet(pl.LightningModule):
    def __init__(self, num_classes, alpha=0.5, gamma=1.0, margin=1.0, emb_dim=512, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.margin = margin
        self.emb_dim = emb_dim

        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batch_size, num_points, _ = x.size()
        x = x.transpose(2, 1)

        input_transform = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), input_transform).transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        feature_transform = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), feature_transform).transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        preds = self(data)
        loss = F.nll_loss(preds, labels)
        self.log('Loss/train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        preds = self(data)
        loss = F.nll_loss(preds, labels)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        self.log('Loss/val_loss', loss, prog_bar=True)
        self.log('Acc/val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            data, labels = batch
            preds = self(data)
            loss = F.nll_loss(preds, labels)
            acc = (preds.argmax(dim=1) == labels).float().mean()
            self.log('Loss/test_loss', loss, prog_bar=True)
            self.log('Acc/test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
