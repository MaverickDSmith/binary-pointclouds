import os
import bitarray
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from binary_encoder import rle_decode_variable_length
from models.CustomCNN import CustomCNN
from dataloaders.BitArrayDataset import BitArrayDataset, PointCloudDataModule

def train_main():
    # Instantiate the DataModule and Model
    # sys.set_int_max_str_digits(274625)
    root_dir = '/home/hi5lab/pointcloud_data/storage_test_two/slice64'
    datamodule = PointCloudDataModule(root_dir)
    datamodule.setup()
    checkpoint_path = '/home/hi5lab/wsl_github/github_ander/Fall 2024/binary-pointclouds/tb_logs/pointcloud_cnn/version_34/checkpoints/epoch=64-Acc/val_acc=0.8704.ckpt'
    checkpoint = False
    _, _, _, num_slices = next(iter(datamodule.train_dataloader()))
    num_slices = num_slices[0].item()

    if checkpoint:
        model = CustomCNN.load_from_checkpoint(checkpoint_path, num_classes=40, num_slices=num_slices)
    else:
        model = CustomCNN(num_classes=40, num_slices=num_slices)

    # Logging and Checkpointing
    logger = TensorBoardLogger("tb_logs", name="pointcloud_cnn")
    checkpoint_callback = ModelCheckpoint(monitor='Acc/val_acc', save_top_k=1, mode='max', filename='{epoch}-{Acc/val_acc:.4f}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Train the model using PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=100,
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

    model_path = "final_model.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Model saved at {model_path}")

def test_main():
    # Step 1: Load the previously saved model
    model_path = "final_model.ckpt"  # Path to your saved model checkpoint
    loaded_model = CustomCNN.load_from_checkpoint(model_path, num_classes=40, num_slices=65)

    # Step 2: Set up the DataModule (same as before)
    root_dir = '/home/hi5lab/pointcloud_data/storage_test_two/slice64'
    datamodule = PointCloudDataModule(root_dir)
    datamodule.setup()
    logger = TensorBoardLogger("tb_logs", name="pointcloud_cnn")


    # Step 3: Create a Trainer instance without training
    trainer = Trainer(
        logger=logger,  # You can reuse the logger if needed
        accelerator='gpu',  # Ensure you're using GPUs
        devices=1,  # Set to the number of GPUs you want to use
        strategy=DDPStrategy(find_unused_parameters=False),  # Enable DDP if using multiple GPUs
        num_nodes=1  # This will be more than 1 when scaling to multiple GPUs
    )

    # Step 4: Run the test step
    trainer.test(loaded_model, datamodule.test_dataloader())


if __name__ == '__main__':
    train_main()
    # test_main()