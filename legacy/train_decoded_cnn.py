import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
import bitarray
import numpy as np
import math
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from binary_encoder import rle_decode_variable_length, decode_binary
import random

from learning3d.models import PointNet, Classifier


class BitArrayDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = []
        self.labels = []
        self.label_to_index = {}
        self.class_dict = {}

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                if label not in self.label_to_index:
                    self.label_to_index[label] = len(self.label_to_index)
                for file_name in os.listdir(label_dir):
                    if file_name.endswith('.bin'):
                        file_path = os.path.join(label_dir, file_name)
                        self.file_paths.append(file_path)
                        label_index = self.label_to_index[label]
                        self.labels.append(label_index)

                        # Add to class_dict for triplet sampling
                        if label_index not in self.class_dict:
                            self.class_dict[label_index] = []
                        self.class_dict[label_index].append(file_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load anchor
        anchor_path = self.file_paths[idx]
        anchor_label = self.labels[idx]
        anchor_tensor, num_slices = self.load_bitarray(anchor_path)
        
        # Load a positive sample from the same class
        positive_idx = self.get_positive_sample(anchor_label)
        positive_path = self.file_paths[positive_idx]
        positive_tensor, _ = self.load_bitarray(positive_path)
        
        # Load a negative sample from a different class
        negative_idx = self.get_negative_sample(anchor_label)
        negative_path = self.file_paths[negative_idx]
        negative_tensor, _ = self.load_bitarray(negative_path)

        # Convert to tensors and return as a tuple
        return (anchor_tensor, anchor_label), \
               (positive_tensor, self.labels[positive_idx]), \
               (negative_tensor, self.labels[negative_idx]), \
               num_slices

    def load_bitarray(self, file_path):
        with open(file_path, 'rb') as f:
            ba = bitarray.bitarray()
            ba.fromfile(f)
        ba, min_bound, max_bound = rle_decode_variable_length(ba)
        size = max_bound - min_bound
        numpy_array_loaded = np.array(ba.tolist(), dtype=np.uint8)
        slices = (round(math.pow(len(ba), 1/3)) - 1)
        grid_points = decode_binary(numpy_array_loaded, slices, size, min_bound)
        # grid_points = grid_points.T # Reshape to (3, num_points)
        return torch.tensor(grid_points, dtype=torch.float32), slices

    def get_positive_sample(self, anchor_label):
        """Get a positive sample index for the given label."""
        positive_indices = [i for i, label in enumerate(self.labels) if label == anchor_label]
        return random.choice(positive_indices)

    def get_negative_sample(self, anchor_label):
        """Get a negative sample index different from the anchor label."""
        negative_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
        return random.choice(negative_indices)
    
    def pad_tensor(self, tensor, max_size):
        # tensor is (num_points, 3)
        current_size = tensor.size(0)  # num_points dimension
        
        if current_size < max_size:
            # Pad along num_points dimension, keep channels (3) intact
            padding = torch.zeros((max_size - current_size, tensor.size(1)), dtype=tensor.dtype)  # Shape (padding_size, 3)
            padded_tensor = torch.cat((tensor, padding), dim=0)  # Concatenate along num_points dimension
            return padded_tensor
        return tensor

    def collate_fn(self, batch):
        # Determine the maximum number of points across anchor, positive, and negative tensors
        max_num_points = max(
            max(item[0][0].size(0), item[1][0].size(0), item[2][0].size(0)) for item in batch
        )

        padded_batch = []
        for (anchor_tensor, anchor_label), (positive_tensor, pos_label), (negative_tensor, neg_label), num_slices in batch:
            # Pad each tensor to the global max_num_points
            padded_anchor = self.pad_tensor(anchor_tensor, max_num_points)
            padded_positive = self.pad_tensor(positive_tensor, max_num_points)
            padded_negative = self.pad_tensor(negative_tensor, max_num_points)
            
            # Append the padded data
            padded_batch.append((padded_anchor, anchor_label, padded_positive, pos_label, padded_negative, neg_label, num_slices))

        # Convert to tensors before returning
        return torch.utils.data.dataloader.default_collate(padded_batch)

    

# LightningModule that handles the model and training/validation loop
class CustomCNN(pl.LightningModule):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # Losses
        self.classification_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

        self.pointnet = PointNet(emb_dims=512, use_bn=True)
        self.classifying = Classifier(feature_model = self.pointnet, num_classes=num_classes)

    def forward(self, x, embeddings=False):

        embedding = self.pointnet(x)
        if embeddings:
            return embedding
        preds = self.classifying(x)
        return embedding, preds
        
        

    def training_step(self, batch, batch_idx):
        # Unpack batch
        anchor_input, anchor_label, positive_input, _, negative_input, _, _ = batch

        # Forward pass
        anchor_output, anchor_preds = self(anchor_input)  # (batch_size, num_classes)
        positive_output = self(positive_input, True)      # (batch_size, embedding_size)
        negative_output = self(negative_input, True)      # (batch_size, embedding_size)
        
        # Classification loss
        loss_classification = self.classification_loss(anchor_preds, anchor_label)

        # Triplet loss
        loss_triplet = self.triplet_loss(anchor_output, positive_output, negative_output)
        
        # Final loss with a weighted sum of classification and triplet loss
        loss = loss_classification + 0.5 * loss_triplet

        # Log the training loss
        self.log('Loss/train_loss', loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        anchor_input, anchor_label, positive_input, _, negative_input, _, _ = batch

        # Forward pass
        anchor_output, anchor_preds = self(anchor_input)  # (batch_size, num_classes)
        positive_output = self(positive_input, True)      # (batch_size, embedding_size)
        negative_output = self(negative_input, True)      # (batch_size, embedding_size)
        
        # Classification loss
        loss_classification = self.classification_loss(anchor_preds, anchor_label)

        # Triplet loss
        loss_triplet = self.triplet_loss(anchor_output, positive_output, negative_output)
        
        # Final loss with a weighted sum of classification and triplet loss
        loss = loss_classification + 0.5 * loss_triplet

        # Calculate accuracy
        preds = torch.argmax(anchor_preds, dim=1)
        acc = (preds == anchor_label).float().mean()

        self.log('Loss/val_loss', loss, prog_bar=True)
        self.log('Acc/val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        anchor_input, anchor_label, positive_input, _, negative_input, _, _ = batch

        # Forward pass
        anchor_output, anchor_preds = self(anchor_input)  # (batch_size, num_classes)
        positive_output = self(positive_input, True)      # (batch_size, embedding_size)
        negative_output = self(negative_input, True)      # (batch_size, embedding_size)
        
        # Classification loss
        loss_classification = self.classification_loss(anchor_preds, anchor_label)

        # Triplet loss
        loss_triplet = self.triplet_loss(anchor_output, positive_output, negative_output)
        
        # Final loss with a weighted sum of classification and triplet loss
        loss = loss_classification + 0.5 * loss_triplet

        # Calculate accuracy
        preds = torch.argmax(anchor_output, dim=1)
        acc = (preds == anchor_label).float().mean()

        self.log('Loss/test_loss', loss, prog_bar=True)
        self.log('Acc/test_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'Loss/val_loss'}



# DataModule that handles loading the data
class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=24, val_size=0.2, test_size=0.1, num_workers=16):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.collate_fn = None

    def setup(self, stage=None):
        dataset = BitArrayDataset(self.root_dir)
        self.collate_fn = dataset.collate_fn
        train_size = int((1 - self.val_size - self.test_size) * len(dataset))
        val_size = int(self.val_size * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn)




# Instantiate the DataModule and Model
root_dir = '/home/hi5lab/pointcloud_data/storage_test_two/slice64'
datamodule = PointCloudDataModule(root_dir)
model = CustomCNN(num_classes=40)

# Logging and Checkpointing
logger = TensorBoardLogger("tb_logs", name="pointcloud_cnn")
checkpoint_callback = ModelCheckpoint(monitor='Acc/val_acc', save_top_k=1, mode='min', filename='{epoch}-{val_loss:.4f}')
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Train the model using PyTorch Lightning Trainer
trainer = Trainer(
    max_epochs=50,
    logger=logger,
    callbacks=[checkpoint_callback, lr_monitor],
    accelerator='gpu',  # Ensure you're using GPUs
    devices=1,  # Set to the number of GPUs you want to use (1 for single-GPU)
    strategy=DDPStrategy(find_unused_parameters=False),  # Enable DDP
    num_nodes=1  # This will be more than 1 when scaling to multiple GPUs
)
trainer.fit(model, datamodule)

# Test the model
trainer.test(model, datamodule.test_dataloader())
